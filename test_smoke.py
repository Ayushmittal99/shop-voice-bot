"""Offline smoke test: real ChromaDB + real MCP server/client + FastAPI, but a fake embedder and
a scripted LLM, so it needs no model downloads and no API keys.   Run: python test_smoke.py"""
import asyncio
import hashlib
import json
import re
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np

# ---- fake sentence_transformers (hashed bag-of-words) -------------------------------------
class FakeST:
    def __init__(self, *a, **k): pass
    def encode(self, items, **k):
        out = []
        for t in items:
            v = np.zeros(8192)
            for w in re.findall(r"[a-z0-9]+", str(t).lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 8192] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)

sys.modules["sentence_transformers"] = types.SimpleNamespace(SentenceTransformer=FakeST)

# ---- point storage at a temp dir BEFORE importing project modules ---------------------------
import config
tmp = Path(tempfile.mkdtemp())
config.CHROMA_DIR, config.LOG_DIR = tmp / "chroma", tmp / "logs"
config.CART_FILE = tmp / "cart.json"
config.SEARCH_MIN_SCORE = 0.05
config.MCP_PORT = 18001
config.MCP_URL = "http://127.0.0.1:18001/mcp"

import catalog, cart, rag, ingest, mcp_server, agent  # noqa: E402
rag.SEARCH_MIN_SCORE = 0.05
cart.CART_FILE = config.CART_FILE
agent.MCP_URL = config.MCP_URL

def ok(msg): print("PASS", msg)

# 1. catalogue + formatting
prods = catalog.load_products()
assert len(prods) == 40
assert catalog.format_price(124999) == "Rs 1,24,999" and catalog.format_price(2499) == "Rs 2,499"
assert catalog.match_value("home appliance", catalog.categories()) == "Home Appliances"
ok("catalogue loads, Indian price format, category matching")

# 2. ingest is incremental
s1 = ingest.ingest(); assert s1["embedded"] == 40, s1
s2 = ingest.ingest(); assert s2["embedded"] == 0 and s2["unchanged"] == 40, s2
ok("ingest: 40 embedded, second run skips all")

# 3. retrieval + filters
r = rag.retrieve("running shoes", max_price=3000)
assert r["products"] and all(p["price"] <= 3000 for p in r["products"]), r
assert r["products"][0]["category"] == "Footwear"
r = rag.retrieve("earbuds with noise cancellation", category="Audio")
assert r["products"][0]["id"] == "P011", [p["id"] for p in r["products"]]
r = rag.retrieve("gaming laptop", max_price=20000)
assert r["relaxed"] == "price" and r["products"], r
r = rag.retrieve("gaming laptop", category="Laptops")            # P009 is out of stock
assert "P009" not in [p["id"] for p in r["products"]]
r = rag.retrieve("gaming laptop", category="Laptops", in_stock_only=False)
assert "P009" in [p["id"] for p in r["products"]]
ok("retrieve: price filter, category filter, budget fallback, stock filter")

# 4. MCP server over HTTP + agent loop with a scripted LLM
import uvicorn
server = uvicorn.Server(uvicorn.Config(mcp_server.mcp.streamable_http_app(), host="127.0.0.1", port=18001, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started: time.sleep(0.1)

class Msg:
    def __init__(s, content=None, calls=None): s.content, s.tool_calls = content, calls
def call(name, args, i=[0]):
    i[0] += 1
    return types.SimpleNamespace(id=f"c{i[0]}", function=types.SimpleNamespace(name=name, arguments=json.dumps(args)))
class Resp:
    def __init__(s, m): s.choices = [types.SimpleNamespace(message=m)]

script, seen = [], []
class FakeGroq:
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                seen.append(kw["messages"]); return Resp(script.pop(0)(kw["messages"]))
agent._get_client = lambda: FakeGroq

async def run():
    async with agent.ShopAgent() as a:
        names = [t["function"]["name"] for t in a.tools]
        assert set(names) == {"search_products", "get_product_details", "compare_products",
                              "add_to_cart", "view_cart", "remove_from_cart", "search_web"}, names
        ok(f"agent sees 7 MCP tools")

        # turn 1: search
        script[:] = [lambda m: Msg(calls=[call("search_products", {"query": "running shoes", "max_price": 3000})]),
                     lambda m: Msg("I found AeroRun Pro for 2,799 rupees.")]
        res = await a.ask("running shoes under 3000")
        assert res["tools_used"] == ["search_products"]
        tool_msg = [m for m in seen[-1] if m["role"] == "tool"][0]["content"]
        assert "[P021]" in tool_msg and "Rs 2,799" in tool_msg, tool_msg
        note = res["history_entries"][1]["content"]
        assert "Products I showed: 1. P0" in note, note
        ok("turn 1: search_products result reached the LLM; 'products shown' note saved to history")
        print("    tool output:\n     ", tool_msg.replace("\n", "\n      ")[:420])

        # turn 2: "add the first one"  (LLM reads the id from the history note)
        hist = res["history_entries"]
        pid = re.search(r"1\. (P\d+)", note).group(1)
        script[:] = [lambda m: Msg(calls=[call("add_to_cart", {"product_id": pid, "quantity": 2})]),
                     lambda m: Msg(calls=[call("view_cart", {})]),
                     lambda m: Msg("Added two pairs. Your total is 5,598 rupees.")]
        res2 = await a.ask("add two of the first one", hist)
        assert res2["tools_used"] == ["add_to_cart", "view_cart"]
        assert "Cart total: Rs 5,598" in cart.summary(), cart.summary()
        ok("turn 2: add_to_cart + view_cart, cart persisted on disk")

        # tools directly: errors, compare, details, limits, removal
        t = lambda n, args: asyncio.ensure_future(a._call_tool(n, args))
        assert "NOT_FOUND" in await a._call_tool("add_to_cart", {"product_id": "zzz"})
        assert "OUT_OF_STOCK" in await a._call_tool("add_to_cart", {"product_id": "P009"})
        assert "LIMIT" in await a._call_tool("add_to_cart", {"product_id": pid, "quantity": 50})
        cmp_ = await a._call_tool("compare_products", {"product_ids": ["P011", "BassBuds Lite"]})
        assert "Cheapest: BassBuds Lite" in cmp_ and "Highest rated: BassBuds Neo" in cmp_, cmp_
        assert "Specs:" in await a._call_tool("get_product_details", {"product_id": "p013"})
        assert "Cart total" not in await a._call_tool("remove_from_cart", {"product_id": "all"})
        assert "empty" in await a._call_tool("view_cart", {})
        assert "SERP_API_KEY" in await a._call_tool("search_web", {"query": "x"})
        assert "Error" in await a._call_tool("search_products", {"query": "x", "max_price": "abc"})
        ok("tools: not-found, out-of-stock, limit, compare by id/name, details, clear cart, web key check, bad args")

        # no match
        out = await a._call_tool("search_products", {"query": "zzzz qqqq"})
        print("    no-match output:", out[:120])
asyncio.run(run())

# 5. FastAPI
from fastapi.testclient import TestClient
import main
main.MCP_URL = config.MCP_URL
c = TestClient(main.app)
assert c.get("/health").json() == {"status": "ok"}
res = c.post("/search", json={"query": "air fryer", "max_price": 6000}).json()["results"]
assert "[P031]" in res, res
assert "empty" in c.get("/cart").json()["cart"]
assert "Added 1 x" in c.post("/cart/add", json={"product_id": "P031"}).json()["cart"]
script[:] = [lambda m: Msg("Hello there.")]
r = c.post("/ask", json={"question": "hi"}).json()
assert r["answer"] == "Hello there." and len(r["history"]) == 2, r
# web UI + structured endpoints
assert "Shop Voice Bot" in c.get("/").text
assert c.get("/static/index.html").status_code == 200
assert c.get("/config").json()["categories"], "no categories"
allp = c.get("/products").json()["products"]
assert len(allp) == 40 and allp[0]["price_text"].startswith("Rs"), allp[0]
assert all(p["category"] == "Kitchen" for p in c.get("/products", params={"category": "Kitchen"}).json()["products"])
assert c.get("/products", params={"q": "air fryer"}).json()["products"]
ci = c.get("/cart/items").json(); assert ci["count"] == 1 and ci["items"][0]["id"] == "P031" and ci["total_text"].startswith("Rs"), ci
assert "Updated the cart" in c.post("/cart/remove", json={"product_id": "P031", "quantity": 0}).json()["cart"]
assert c.get("/cart/items").json()["count"] == 0
# /ask returns product cards + cart_changed for the UI
script[:] = [lambda m: Msg(calls=[call("search_products", {"query": "air fryer", "max_price": 6000})]),
             lambda m: Msg("Here are air fryers.")]
r = c.post("/ask", json={"question": "air fryers"}).json()
assert r["products"] and r["products"][0]["id"] == "P031" and r["cart_changed"] is False, r
script[:] = [lambda m: Msg(calls=[call("add_to_cart", {"product_id": "P031"})]), lambda m: Msg("Added.")]
r = c.post("/ask", json={"question": "add it"}).json()
assert r["cart_changed"] is True and c.get("/cart/items").json()["count"] == 1, r
c.post("/cart/remove", json={"product_id": "all"})
# /transcribe (Whisper call faked)
import voice
voice.speech_to_text = lambda data, name="question.wav": "running shoes"
assert c.post("/transcribe", files={"audio": ("q.webm", b"x" * 5000, "audio/webm")}).json() == {"text": "running shoes"}
assert c.post("/transcribe", files={"audio": ("q.webm", b"x" * 10, "audio/webm")}).json() == {"text": ""}
ok("FastAPI: web UI, /products /cart/items /cart/remove /transcribe, /health /search /cart /cart/add /ask")

# 6. speech cleaning
from voice import clean_for_speech
s = clean_for_speech("Try the **AeroRun Pro** [P021] for Rs 2,799 (MRP Rs 3,999).")
assert s == "Try the AeroRun Pro for 2,799 rupees (M R P 3,999 rupees).", s
ok(f"voice text cleaning -> '{s}'")

# 6b. the real voice_chat loop (typed input, mute, exit)
import builtins, io, contextlib, voice, voice_chat
inputs = iter(["running shoes under 3000", "mute", "exit"])
builtins.input = lambda p="": next(inputs)
spoken = []
voice.text_to_speech = lambda text: spoken.append(text) or 0.01
script[:] = [lambda m: Msg(calls=[call("search_products", {"query": "running shoes", "max_price": 3000})]),
             lambda m: Msg("AeroRun Pro running shoes are 2,799 rupees. Want them?")]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    asyncio.run(voice_chat.main())
out = buf.getvalue()
assert "Bot: AeroRun Pro" in out and "Tools: search_products" in out and "Spoken answers off" in out, out
assert spoken == ["AeroRun Pro running shoes are 2,799 rupees. Want them?"], spoken
ok("voice_chat loop: typed request -> agent -> printed + spoken answer; mute works")

# 7. MCP server down -> friendly error
agent.MCP_URL = "http://127.0.0.1:1/mcp"
async def down():
    try:
        async with agent.ShopAgent("http://127.0.0.1:1/mcp"): pass
    except agent.MCPUnavailable as e:
        return str(e)
msg = asyncio.run(down()); assert msg and "python mcp_server.py" in msg, msg
ok("server down -> MCPUnavailable with start instructions")
print("\nALL TESTS PASSED")
