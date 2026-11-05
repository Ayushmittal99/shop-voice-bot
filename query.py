"""Debug retrieval without the LLM.

Usage: python query.py "wireless earbuds" [--max-price 3000] [--category Audio] [--brand Boomtone] [-k 5]
"""
import argparse

from catalog import format_price
from config import SEARCH_TOP_K
from rag import retrieve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--max-price", type=float, default=0)
    ap.add_argument("--min-price", type=float, default=0)
    ap.add_argument("--category", default="")
    ap.add_argument("--brand", default="")
    ap.add_argument("--min-rating", type=float, default=0)
    ap.add_argument("--include-out-of-stock", action="store_true")
    ap.add_argument("-k", type=int, default=SEARCH_TOP_K)
    a = ap.parse_args()

    r = retrieve(
        a.query, a.max_price, a.min_price, a.category, a.brand, a.min_rating,
        not a.include_out_of_stock, a.k,
    )
    print(f"Filters: { {k: v for k, v in r['filters'].items() if v} }")
    if r["relaxed"]:
        print(f"(nothing matched the {r['relaxed']} filter, showing the closest products instead)")
    for p in r["products"]:
        print(f"  score={p['score']:.3f}  [{p['id']}] {p['name']}  {format_price(p['price'])}  "
              f"{p['rating']}*  stock={p['stock']}")
    if not r["products"]:
        print("  no matches")
    print(f"\n{r['seconds']}s")
