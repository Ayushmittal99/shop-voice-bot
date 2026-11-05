"""Usage: python ingest.py [path/to/products.json|csv] [--force]

Embeds the product catalogue into ChromaDB. Incremental: only new or changed products are
embedded, and products removed from the catalogue are removed from the index.
"""
import argparse
import time
from pathlib import Path

from catalog import content_hash, image_path, load_products, product_text
from config import CATALOG_FILE
from embedder import embed_images, embed_texts
from logger_setup import get_logger, log_event
from vector_store import VectorStore

logger = get_logger()


def ingest(catalog_path: Path = CATALOG_FILE, force: bool = False) -> dict:
    start = time.time()
    log_event(logger, "Ingestion started", file=catalog_path.name, force=force)
    try:
        products = load_products(catalog_path)
        store = VectorStore()
        stored = store.stored_hashes()
        hashes = {pid: content_hash(p) for pid, p in products.items()}

        removed = [pid for pid in stored if pid not in products]
        changed = [pid for pid in products if force or stored.get(pid) != hashes[pid]]
        store.delete(removed + ([pid for pid in changed if pid in stored]))  # clears stale image rows too

        images_done = 0
        if changed:
            items = [products[pid] for pid in changed]

            # 1. text embeddings (one product = one vector)
            vectors = embed_texts([product_text(p) for p in items])
            store.add_products(
                ids=[p["id"] for p in items],
                embeddings=vectors,
                documents=[product_text(p) for p in items],
                metadatas=[
                    {
                        "product_id": p["id"], "name": p["name"], "brand": p["brand"],
                        "category": p["category"], "price": p["price"], "rating": p["rating"],
                        "stock": p["stock"], "content_hash": hashes[p["id"]],
                    }
                    for p in items
                ],
            )
            log_event(logger, "Products stored", count=len(items))

            # 2. optional image embeddings (only products whose image file exists)
            with_img = [(p, image_path(p)) for p in items if image_path(p)]
            if with_img:
                vecs = embed_images([str(path) for _, path in with_img])
                store.add_images(
                    ids=[p["id"] for p, _ in with_img],
                    embeddings=vecs,
                    documents=[p["name"] for p, _ in with_img],
                    metadatas=[
                        {"product_id": p["id"], "category": p["category"], "price": p["price"],
                         "stock": p["stock"], "image_path": str(path)}
                        for p, path in with_img
                    ],
                )
                images_done = len(with_img)
                log_event(logger, "Product images stored", count=images_done)

        summary = {
            "status": "ok", "products_in_catalogue": len(products), "embedded": len(changed),
            "removed": len(removed), "unchanged": len(products) - len(changed),
            "images": images_done, "seconds": round(time.time() - start, 2),
        }
        log_event(logger, "Ingestion finished", **summary)
        return summary
    except Exception:
        logger.exception("Ingestion failed")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("catalog", type=Path, nargs="?", default=CATALOG_FILE)
    ap.add_argument("--force", action="store_true", help="re-embed every product")
    args = ap.parse_args()
    print(ingest(args.catalog, force=args.force))
