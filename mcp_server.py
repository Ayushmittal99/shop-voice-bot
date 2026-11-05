
"""
MCP server (port 8001) - shop tools with RAG + Google Shopping.

Tools:
1. search_products      : semantic search in the local store catalogue
2. get_product_details  : full details of a local product
3. compare_products     : compare 2-4 local products
4. add_to_cart          : add a local product to cart
5. view_cart            : show cart items and total
6. remove_from_cart     : remove a local product from cart
7. search_web           : Google Shopping search using SerpAPI

Run:
    python mcp_server.py

MCP endpoint:
    http://127.0.0.1:8001/mcp
"""

import os

import requests
from mcp.server.fastmcp import FastMCP

import cart
from catalog import (
    categories,
    find_products,
    format_price,
    load_products,
    product_detail,
    product_line,
)
from config import MCP_HOST, MCP_PORT, SEARCH_TOP_K
from rag import retrieve


mcp = FastMCP("shop-server", host=MCP_HOST, port=MCP_PORT)


# ---------------- Shared helper ----------------

def _one(ref: str) -> tuple[dict | None, str]:
    """Resolve a product ID or name to exactly one local product."""

    matches = find_products(ref)

    if len(matches) == 1:
        return matches[0], ""

    if not matches:
        return (
            None,
            f"NOT_FOUND: no product matches '{ref}'. "
            "Use search_products to find the right one.",
        )

    names = "; ".join(
        f"[{p['id']}] {p['name']}" for p in matches[:5]
    )

    return (
        None,
        f"AMBIGUOUS: '{ref}' matches several products: {names}. "
        "Ask which one.",
    )


# ---------------- Tool 1: search_products (local RAG) ----------------

@mcp.tool(
    description=(
        "Search the local store catalogue using semantic product search. "
        "Use for products stocked by our store. "
        "query describes the product and desired features. "
        "Optional filters: max_price, min_price in rupees, category, brand, "
        "min_rating from 0-5, in_stock_only and limit. "
        "For online products outside our catalogue, use search_web."
    )
)
def search_products(
    query: str,
    max_price: float = 0,
    min_price: float = 0,
    category: str = "",
    brand: str = "",
    min_rating: float = 0,
    in_stock_only: bool = True,
    limit: int = SEARCH_TOP_K,
) -> str:
    """Search products in the local catalogue using RAG."""

    try:
        r = retrieve(
            query,
            max_price,
            min_price,
            category,
            brand,
            min_rating,
            in_stock_only,
            k=max(1, min(int(limit), 8)),
        )
    except RuntimeError as e:
        return f"Error: {e}"

    filters = r["filters"]

    applied = [f"query '{r['query']}'"]

    if filters["max_price"]:
        applied.append(
            f"max price {format_price(filters['max_price'])}"
        )

    if filters["min_price"]:
        applied.append(
            f"min price {format_price(filters['min_price'])}"
        )

    for key in ("category", "brand"):
        if filters[key]:
            applied.append(f"{key} {filters[key]}")

    if filters["min_rating"]:
        applied.append(f"rating >= {filters['min_rating']}")

    header = "Filters: " + ", ".join(applied)

    if not r["products"]:
        ignored = [
            name
            for name, given in (
                ("category", category),
                ("brand", brand),
            )
            if given and not filters[name]
        ]

        hint = (
            f" (unknown {', '.join(ignored)} ignored)"
            if ignored
            else ""
        )

        return (
            f"NO_MATCH. {header}{hint}. "
            f"Categories we sell: {', '.join(categories())}. "
            "Try search_web to find online alternatives."
        )

    lines = [
        product_line(product, i)
        for i, product in enumerate(r["products"], 1)
    ]

    top = r["products"][0]["description"]

    if r["relaxed"] == "price":
        return (
            f"NO_MATCH_IN_BUDGET. {header}. "
            "Nothing matches that price. "
            "Closest products WITHOUT the price limit:\n"
            + "\n".join(lines)
        )

    return (
        f"{header}\n"
        f"Found {len(lines)} local product(s):\n"
        + "\n".join(lines)
        + f"\nTop match: {top}"
    )


# ---------------- Tool 2: get_product_details ----------------

@mcp.tool()
def get_product_details(product_id: str) -> str:
    """
    Get full details of a local product, including price, MRP,
    rating, stock, description and specifications.
    Accepts a product ID such as P012 or a product name.
    """

    product, error = _one(product_id)

    if not product:
        return error

    return product_detail(product)


# ---------------- Tool 3: compare_products ----------------

@mcp.tool()
def compare_products(product_ids: list[str]) -> str:
    """
    Compare 2-4 products from the local store catalogue.
    Example: ['P011', 'P013'].
    """

    if not 2 <= len(product_ids) <= 4:
        return "Error: give between 2 and 4 product IDs."

    items = []

    for ref in product_ids:
        product, error = _one(ref)

        if not product:
            return error

        items.append(product)

    blocks = []

    for product in items:
        specs = "; ".join(
            f"{key}: {value}"
            for key, value in product["specs"].items()
        )

        blocks.append(
            f"{product_line(product)}\n   {specs}"
        )

    cheapest = min(items, key=lambda p: p["price"])

    best_rated = max(
        items,
        key=lambda p: (p["rating"], p["reviews"]),
    )

    return (
        "\n".join(blocks)
        + f"\nCheapest: {cheapest['name']} "
          f"({format_price(cheapest['price'])})."
        + f"\nHighest rated: {best_rated['name']} "
          f"({best_rated['rating']} stars)."
    )


# ---------------- Tool 4: add_to_cart ----------------

@mcp.tool()
def add_to_cart(product_id: str, quantity: int = 1) -> str:
    """
    Add a local store product to the cart.
    Only call when the customer clearly requests adding or buying it.
    """

    product, error = _one(product_id)

    if not product:
        return error

    return cart.add(product["id"], quantity)


# ---------------- Tool 5: view_cart ----------------

@mcp.tool()
def view_cart() -> str:
    """Show local cart items and the total price."""

    return cart.summary()


# ---------------- Tool 6: remove_from_cart ----------------

@mcp.tool()
def remove_from_cart(
    product_id: str,
    quantity: int = 0,
) -> str:
    """
    Remove or reduce a local product in the cart.
    quantity=0 removes it completely.
    product_id='all' empties the cart.
    """

    if product_id.strip().lower() == "all":
        return cart.clear()

    product, error = _one(product_id)

    if not product:
        return error

    return cart.remove(product["id"], quantity)


# ---------------- Tool 7: Google Shopping (SerpAPI) ----------------

@mcp.tool(
    description=(
        "Search Google Shopping for online products available in India. "
        "Use this when customers want products from across the internet, "
        "more options than our local catalogue, online prices, sellers, "
        "ratings, or product links. "
        "Returns external shopping listings with prices, images and URLs "
        "when available. These are not local store products and are not "
        "automatically added to the local cart. "
        "Use search_products first for items in our own catalogue; "
        "use search_web for online shopping results."
    )
)
def search_web(query: str) -> str:
    """
    Search Google Shopping for products in India using SerpAPI.

    Returns product titles, prices, sellers, ratings, reviews,
    product links and thumbnail URLs when available.
    """

    serp_key = os.getenv("SERP_API_KEY")

    if not serp_key:
        return (
            "Error: SERP_API_KEY is not set. "
            "Add it to your .env file and restart the MCP server."
        )

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_shopping",
                "q": query,
                "google_domain": "google.co.in",
                "gl": "in",
                "hl": "en",
                "api_key": serp_key,
            },
            timeout=30,
        )

        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            # Do not print the request URL or API key.
            return f"SerpAPI error: {data['error']}"

        results = data.get("shopping_results", [])

    except requests.RequestException as e:
        return (
            f"Error: Google Shopping request failed "
            f"({type(e).__name__}). "
            "Check your internet connection, API key and usage limits."
        )

    except ValueError:
        return "Error: SerpAPI returned invalid JSON."

    if not results:
        return (
            f"NO_WEB_RESULTS: No Google Shopping products found for "
            f"'{query}'. Try a broader product search."
        )

    lines = [
        f"GOOGLE SHOPPING RESULTS: {query}",
        "Region: India",
        "Source: External online listings, not our local catalogue.",
        f"Results returned: {len(results)}",
        "",
    ]

    for index, item in enumerate(results, start=1):
        title = item.get("title") or "Unknown product"
        price = item.get("price") or "Price unavailable"
        seller = item.get("source") or "Seller unavailable"
        rating = item.get("rating")
        reviews = item.get("reviews")

        product_url = (
            item.get("product_link")
            or item.get("link")
            or "Not available"
        )

        image_url = (
            item.get("thumbnail")
            or "Not available"
        )

        lines.append(
            f"{index}. {title}\n"
            f"   Price: {price}\n"
            f"   Seller: {seller}\n"
            f"   Rating: {rating if rating is not None else 'Not available'}\n"
            f"   Reviews: {reviews if reviews is not None else 'Not available'}\n"
            f"   Product URL: {product_url}\n"
            f"   Image URL: {image_url}"
        )

    lines.append(
        "\nNote: Prices, ratings and availability may change. "
        "Verify details on the seller's website. "
        "External listings cannot be added to the local store cart "
        "unless a separate integration is implemented."
    )

    return "\n".join(lines)


# ---------------- Start MCP server ----------------

if __name__ == "__main__":
    load_products()  # Fail early if the local catalogue is broken.

    try:
        from embedder import embed_texts

        embed_texts(["warm up"])
    except Exception as e:
        print(f"(model warm-up skipped: {type(e).__name__})")

    print(
        f"Starting shop MCP server at "
        f"http://{MCP_HOST}:{MCP_PORT}/mcp"
    )

    mcp.run(transport="streamable-http")
