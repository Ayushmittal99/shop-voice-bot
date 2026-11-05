"""Tiny persistent shopping cart (data/cart.json). One cart for the whole demo store.
For real users, key this by user/session id and store it in a database."""
import json
import os
import threading

from catalog import format_price, get_product
from config import CART_FILE

_lock = threading.Lock()
MAX_PER_ITEM = 10


def _load() -> dict[str, int]:
    try:
        return {k: int(v) for k, v in json.loads(CART_FILE.read_text(encoding="utf-8")).items()}
    except (FileNotFoundError, ValueError):
        return {}


def _save(cart: dict[str, int]):
    CART_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CART_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cart, indent=2), encoding="utf-8")
    os.replace(tmp, CART_FILE)  # atomic, so a crash can't leave a half-written cart


def summary(cart: dict[str, int] | None = None) -> str:
    cart = _load() if cart is None else cart
    lines, total = [], 0.0
    for pid, qty in cart.items():
        p = get_product(pid)
        if not p:
            continue
        sub = p["price"] * qty
        total += sub
        lines.append(f"- {qty} x [{p['id']}] {p['name']} @ {format_price(p['price'])} = {format_price(sub)}")
    if not lines:
        return "The cart is empty."
    return "\n".join(lines) + f"\nCart total: {format_price(total)}"


def items() -> dict:
    """Cart as data for the web UI: {"items": [{product..., "quantity", "subtotal"}], "count", "total", "total_text"}."""
    rows, total, count = [], 0.0, 0
    for pid, qty in _load().items():
        p = get_product(pid)
        if not p:
            continue
        sub = p["price"] * qty
        total += sub
        count += qty
        rows.append({**p, "quantity": qty, "subtotal": sub, "price_text": format_price(p["price"]),
                     "subtotal_text": format_price(sub)})
    return {"items": rows, "count": count, "total": total, "total_text": format_price(total)}


def add(product_id: str, quantity: int = 1) -> str:
    p = get_product(product_id)
    if not p:
        return f"NOT_FOUND: no product with id {product_id}."
    quantity = max(1, int(quantity))
    with _lock:
        cart = _load()
        new_qty = cart.get(p["id"], 0) + quantity
        if p["stock"] <= 0:
            return f"OUT_OF_STOCK: {p['name']} is currently out of stock."
        if new_qty > min(p["stock"], MAX_PER_ITEM):
            return (f"LIMIT: cannot hold {new_qty} of {p['name']} "
                    f"(stock {p['stock']}, max {MAX_PER_ITEM} per order). Cart has {cart.get(p['id'], 0)}.")
        cart[p["id"]] = new_qty
        _save(cart)
        return f"Added {quantity} x {p['name']}.\n{summary(cart)}"


def remove(product_id: str, quantity: int = 0) -> str:
    """quantity 0 = remove the item completely."""
    p = get_product(product_id)
    pid = p["id"] if p else product_id
    with _lock:
        cart = _load()
        if pid not in cart:
            return f"NOT_IN_CART: {product_id} is not in the cart."
        left = cart[pid] - quantity if quantity > 0 else 0
        if left > 0:
            cart[pid] = left
        else:
            del cart[pid]
        _save(cart)
        return f"Updated the cart.\n{summary(cart)}"


def clear() -> str:
    with _lock:
        _save({})
    return "The cart is now empty."
