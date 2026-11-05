"""RAG retrieval for products: semantic search + metadata filters + live catalogue check.

    question -> embed -> ChromaDB (filtered by price / category / brand / rating / stock)
             -> drop weak matches (SEARCH_MIN_SCORE, RELATIVE_SCORE) -> re-check filters against the catalogue
             -> products (+ scores)

The 'G' (generation) happens in agent.py: the LLM reads these products through the MCP tool
`search_products` and answers the customer using only them.
"""
import time

from catalog import brands, categories, get_product, match_value
from config import CANDIDATE_FACTOR, RELATIVE_SCORE, SEARCH_MIN_SCORE, SEARCH_TOP_K
from embedder import embed_texts
from logger_setup import get_logger, log_event
from vector_store import VectorStore

logger = get_logger()


def build_where(max_price=0, min_price=0, category=None, brand=None, min_rating=0, in_stock_only=True):
    conds = []
    if max_price:
        conds.append({"price": {"$lte": float(max_price)}})
    if min_price:
        conds.append({"price": {"$gte": float(min_price)}})
    if category:
        conds.append({"category": category})
    if brand:
        conds.append({"brand": brand})
    if min_rating:
        conds.append({"rating": {"$gte": float(min_rating)}})
    if in_stock_only:
        conds.append({"stock": {"$gt": 0}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _passes(p: dict, f: dict) -> bool:
    """Re-check filters on the live catalogue, in case price/stock changed after ingestion."""
    return (
        (not f["max_price"] or p["price"] <= f["max_price"])
        and (not f["min_price"] or p["price"] >= f["min_price"])
        and (not f["category"] or p["category"] == f["category"])
        and (not f["brand"] or p["brand"] == f["brand"])
        and (not f["min_rating"] or p["rating"] >= f["min_rating"])
        and (not f["in_stock_only"] or p["stock"] > 0)
    )


def _search(store: VectorStore, query: str, f: dict, k: int) -> list[dict]:
    emb = embed_texts([query])[0]

    # How relevant is the best buyable product for this query at all? (category/brand/stock
    # applied, but NOT price/rating.) Hits must be close to it, otherwise a tight budget would
    # "match" unrelated products that merely passed the price filter.
    ref = store.search(emb, k=1, where=build_where(
        category=f["category"], brand=f["brand"], in_stock_only=f["in_stock_only"]))
    ref_best = 1 - ref["distances"][0][0] if ref["distances"][0] else 0.0
    cutoff = max(SEARCH_MIN_SCORE, RELATIVE_SCORE * ref_best)

    res = store.search(emb, k=k * CANDIDATE_FACTOR, where=build_where(**f))
    hits = []
    for pid, dist in zip(res["ids"][0], res["distances"][0]):
        score = 1 - dist
        p = get_product(pid)
        if p and score >= cutoff and _passes(p, f):
            hits.append({**p, "score": round(score, 3)})
    hits.sort(key=lambda h: h["score"], reverse=True)
    return hits[:k]


def retrieve(
    query: str,
    max_price: float = 0,
    min_price: float = 0,
    category: str = "",
    brand: str = "",
    min_rating: float = 0,
    in_stock_only: bool = True,
    k: int = SEARCH_TOP_K,
) -> dict:
    """Returns {"products": [...], "filters": {...}, "relaxed": None | "price", "seconds": float}."""
    t0 = time.perf_counter()
    cat = match_value(category, categories()) if category else None
    br = match_value(brand, brands()) if brand else None
    f = {
        "max_price": max_price or 0, "min_price": min_price or 0, "category": cat,
        "brand": br, "min_rating": min_rating or 0, "in_stock_only": in_stock_only,
    }
    query = (query or "").strip() or cat or br or "popular products"

    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError("The product index is empty. Run `python ingest.py` first.")

    products = _search(store, query, f, k)
    relaxed = None
    if not products and (f["max_price"] or f["min_price"]):
        # nothing in budget: find what is closest, so the bot can say "nothing under X, but ..."
        products = _search(store, query, {**f, "max_price": 0, "min_price": 0}, k)
        relaxed = "price" if products else None

    seconds = round(time.perf_counter() - t0, 3)
    log_event(
        logger, "Product search", query=query, filters={k_: v for k_, v in f.items() if v},
        results=len(products), relaxed=relaxed,
        best_score=products[0]["score"] if products else None, seconds=seconds,
    )
    return {"products": products, "filters": f, "relaxed": relaxed, "query": query, "seconds": seconds}
