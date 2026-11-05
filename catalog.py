"""Product catalogue: loading, validation and display helpers.

This replaces pdf_parser.py. One product = one retrievable unit (no chunking needed).
The catalogue file stays the single source of truth for price and stock; ChromaDB is only
used to FIND products. Re-run `python ingest.py` after editing the catalogue.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

from config import CATALOG_FILE, CURRENCY, DATA_DIR
from logger_setup import get_logger, log_event

logger = get_logger()

REQUIRED = ("id", "name", "category", "price")
_cache: dict = {"key": None, "products": {}}


# ---------------------------------------------------------------- loading
def _parse_specs(value) -> dict:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    out = {}
    for part in str(value or "").split(";"):          # CSV form: "RAM: 8GB; Storage: 128GB"
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _parse_tags(value) -> list[str]:
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    return [t.strip() for t in str(value or "").split("|") if t.strip()]


def _normalize(raw: dict) -> dict | None:
    if any(not str(raw.get(f, "")).strip() for f in REQUIRED):
        logger.warning(f"Skipping product with missing fields {REQUIRED}: {raw.get('id') or raw}")
        return None
    try:
        price = float(raw["price"])
        return {
            "id": str(raw["id"]).strip(),
            "name": str(raw["name"]).strip(),
            "brand": str(raw.get("brand", "")).strip(),
            "category": str(raw["category"]).strip(),
            "price": price,
            "mrp": float(raw.get("mrp") or price),
            "rating": float(raw.get("rating") or 0),
            "reviews": int(float(raw.get("reviews") or 0)),
            "stock": int(float(raw.get("stock") or 0)),
            "description": str(raw.get("description", "")).strip(),
            "specs": _parse_specs(raw.get("specs")),
            "tags": _parse_tags(raw.get("tags")),
            "image": str(raw.get("image", "")).strip(),
        }
    except (TypeError, ValueError):
        logger.warning(f"Skipping product with a bad number: {raw.get('id')}")
        return None


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["products"] if isinstance(data, dict) else data


def load_products(path: Path = CATALOG_FILE) -> dict[str, dict]:
    """Returns {product_id: product}. Re-reads the file only when it changed."""
    stat = path.stat()
    key = (str(path), stat.st_mtime, stat.st_size)
    if _cache["key"] == key:
        return _cache["products"]

    products: dict[str, dict] = {}
    for raw in _read_rows(path):
        p = _normalize(raw)
        if p:
            if p["id"] in products:
                logger.warning(f"Duplicate product id {p['id']}, keeping the last one")
            products[p["id"]] = p
    _cache.update(key=key, products=products)
    log_event(logger, "Catalogue loaded", file=path.name, products=len(products))
    return products


# ---------------------------------------------------------------- lookups
def get_product(product_id: str) -> dict | None:
    products = load_products()
    if product_id in products:
        return products[product_id]
    low = product_id.strip().lower()
    return next((p for pid, p in products.items() if pid.lower() == low), None)


def find_products(ref: str) -> list[dict]:
    """Find by id first, then by name words. The voice LLM sometimes passes a name instead of an id."""
    p = get_product(ref)
    if p:
        return [p]
    words = re.findall(r"\w+", ref.lower())
    if not words:
        return []
    return [p for p in load_products().values() if all(w in p["name"].lower() for w in words)]


def categories() -> list[str]:
    return sorted({p["category"] for p in load_products().values()})


def brands() -> list[str]:
    return sorted({p["brand"] for p in load_products().values() if p["brand"]})


def match_value(text: str, options: list[str]) -> str | None:
    """Map what the LLM typed ('home appliance', 'AUDIO') to a real category / brand name."""
    t = (text or "").strip().lower()
    if len(t) < 3:
        return None
    for o in options:
        if o.lower() == t:
            return o
    for o in options:
        if o.lower() in t or t in o.lower():
            return o
    return None


# ---------------------------------------------------------------- text for embedding
def product_text(p: dict) -> str:
    """The text that gets embedded. Anything a customer might say should appear here."""
    specs = "; ".join(f"{k}: {v}" for k, v in p["specs"].items())
    return (
        f"{p['name']}. Brand: {p['brand']}. Category: {p['category']}. "
        f"{p['description']} Specs: {specs}. Tags: {', '.join(p['tags'])}."
    )


def content_hash(p: dict) -> str:
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def image_path(p: dict) -> Path | None:
    if not p["image"]:
        return None
    path = Path(p["image"])
    path = path if path.is_absolute() else DATA_DIR / path
    return path if path.exists() else None


# ---------------------------------------------------------------- display
def format_price(value: float) -> str:
    """Indian digit grouping: 124999 -> 'Rs 1,24,999'."""
    s = str(int(round(value)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        s = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head) + "," + tail
    return f"{CURRENCY} {s}"


def stock_text(p: dict) -> str:
    if p["stock"] <= 0:
        return "out of stock"
    return f"only {p['stock']} left" if p["stock"] <= 5 else "in stock"


def product_line(p: dict, index: int | None = None) -> str:
    """One compact line. The leading '1. [P001] Name |' shape is parsed by agent.py."""
    prefix = f"{index}. " if index is not None else ""
    return (
        f"{prefix}[{p['id']}] {p['name']} | {p['brand']} | {format_price(p['price'])} | "
        f"{p['rating']} stars ({p['reviews']}) | {stock_text(p)}"
    )


def product_detail(p: dict) -> str:
    off = round((1 - p["price"] / p["mrp"]) * 100) if p["mrp"] > p["price"] else 0
    price = format_price(p["price"]) + (f" (MRP {format_price(p['mrp'])}, {off}% off)" if off else "")
    specs = "; ".join(f"{k}: {v}" for k, v in p["specs"].items()) or "n/a"
    lines = [
        f"[{p['id']}] {p['name']} by {p['brand']}",
        f"Category: {p['category']} | Price: {price}",
        f"Rating: {p['rating']} stars from {p['reviews']} reviews | Stock: {stock_text(p)}",
        p["description"],
        f"Specs: {specs}",
    ]
    img = image_path(p)
    if img:
        lines.append(f"Image: {img}")
    return "\n".join(lines)
