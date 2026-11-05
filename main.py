"""FastAPI (port 8000): the web UI + JSON API. Talks to the MCP server (8001) through the shopping agent.

Run:  python mcp_server.py   (terminal 1)
      uvicorn main:app --port 8000   (terminal 2)   ->  open http://localhost:8000
"""
import asyncio
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, Field

import cart
from agent import MCPUnavailable, ShopAgent
from catalog import categories, format_price, get_product, load_products
from config import CURRENCY, MCP_URL

WEB_DIR = Path(__file__).parent / "web"
MAX_AUDIO_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Shop Voice Bot API")


class AskRequest(BaseModel):
    question: str
    history: list[dict] = Field(default_factory=list)  # send back what the last reply returned


class SearchRequest(BaseModel):
    query: str
    max_price: float = 0
    min_price: float = 0
    category: str = ""
    brand: str = ""
    min_rating: float = 0


class CartItem(BaseModel):
    product_id: str
    quantity: int = 1


def card(p: dict) -> dict:
    """A product as the web UI needs it."""
    return {**p, "price_text": format_price(p["price"]), "mrp_text": format_price(p["mrp"]) if p.get("mrp") else ""}


async def call_tool(name: str, args: dict) -> str:
    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, args)
                return result.content[0].text
    except Exception as e:
        raise HTTPException(503, f"MCP server not reachable ({type(e).__name__}). Run: python mcp_server.py")


# ---------------- web UI ----------------
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def ui_config():
    return {"currency": CURRENCY, "categories": categories()}


@app.get("/products")
async def products(category: str = "", q: str = ""):
    """Browse the whole catalogue (no LLM, no search index needed)."""
    q = q.strip().lower()
    rows = [
        p for p in load_products().values()
        if (not category or p["category"].lower() == category.lower())
        and (not q or q in f"{p['name']} {p['brand']} {p['category']} {' '.join(p['tags'])}".lower())
    ]
    return {"products": [card(p) for p in rows]}


# ---------------- assistant ----------------
@app.post("/ask")
async def ask(body: AskRequest):
    """Natural-language shopping request. The agent picks the tools (catalogue search, cart, web...)."""
    try:
        async with ShopAgent() as agent:
            result = await agent.ask(body.question, body.history)
    except MCPUnavailable as e:
        raise HTTPException(503, str(e))
    return {
        "answer": result["answer"],
        "tools_used": result["tools_used"],
        "products": [card(p) for p in map(get_product, result["product_ids"]) if p],
        "cart_changed": result["cart_changed"],
        "history": body.history + result["history_entries"],  # send this back with the next question
    }


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Speech to text for the browser microphone (Groq Whisper)."""
    data = await audio.read()
    if len(data) < 1000:
        return {"text": ""}
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Recording too long.")
    from voice import speech_to_text

    name = audio.filename or "question.webm"
    try:
        text = await asyncio.to_thread(speech_to_text, data, name)
    except Exception as e:
        raise HTTPException(502, f"Speech-to-text failed ({type(e).__name__}). Check GROQ_API_KEY.")
    return {"text": text}


@app.post("/search")
async def search(body: SearchRequest):
    """Direct product search, no LLM (handy for testing and for a product-grid UI)."""
    return {"results": await call_tool("search_products", body.model_dump())}


# ---------------- cart ----------------
@app.get("/cart")
async def view_cart():
    return {"cart": await call_tool("view_cart", {})}


@app.get("/cart/items")
async def cart_items():
    return cart.items()


@app.post("/cart/add")
async def add_to_cart(body: CartItem):
    return {"cart": await call_tool("add_to_cart", body.model_dump())}


@app.post("/cart/remove")
async def remove_from_cart(body: CartItem):
    """quantity 0 removes the item completely; product_id 'all' empties the cart."""
    return {"cart": await call_tool("remove_from_cart", {"product_id": body.product_id, "quantity": max(0, body.quantity)})}


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
