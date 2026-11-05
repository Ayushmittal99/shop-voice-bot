# Shop Voice Bot (RAG + MCP)

A voice shopping assistant. Say *"I need running shoes under 3000 rupees"*, hear the best
matches, then *"add the second one to my cart"*. It is the PDF-RAG project reshaped for e-commerce:
same stack (ChromaDB, MiniLM/CLIP, Groq, MCP, FastAPI, Whisper, TTS), different domain.

## How it works

```
 mic -> Whisper (Groq) -> text
                            |
                       agent.py  (Groq LLM + function calling)
                            |  decides which tool to call
                            v
                  MCP server (mcp_server.py, port 8001)
     search_products  get_product_details  compare_products
     add_to_cart  view_cart  remove_from_cart  search_web
            |                                      |
   rag.py: embed query -> ChromaDB                SerpAPI
   + filters (price/category/brand/rating/stock)
            |
   products come back as text -> LLM answers using ONLY those -> TTS speaks it
```

The LLM turns speech into filters ("under 3000" -> `max_price=3000`), the MCP tool does the
retrieval, and the LLM writes a short spoken answer from the real results. It cannot invent
products, prices or stock because it only sees what the tools return.

### What changed from the PDF project

| PDF project | Shop project |
|---|---|
| `pdf_parser.py`, `chunker.py` | `catalog.py` (one product = one retrievable unit, no chunking) |
| `ingest.py` (per PDF) | `ingest.py` (per catalogue, **incremental**: only changed products are re-embedded) |
| `vector_store.py` (`pdf_text`, `pdf_images`) | `vector_store.py` (`products`, `product_images`) |
| `query.py`, `rag.py` | `query.py` (debug CLI), `rag.py` (`retrieve()` with metadata filters + budget fallback) |
| `generator.py` fixed RAG prompt | `agent.py` LLM tool-calling loop over MCP; `generator.py` is just the Groq client |
| MCP: `ask_pdf`, `summarize_pdf`, `search_web` | MCP: 7 shop tools (see below) |
| `main.py` `/ask`, `/summarize`, `/search` | `main.py` `/ask`, `/search`, `/cart`, `/cart/add` |
| `voice_chat.py` called `rag.answer` | `voice_chat.py` talks to the agent, which calls MCP tools |
| `chat.py` | merged into `voice_chat.py` (you can type or speak) |
| `usage.py`, `usage_report.py` | **removed** (no API call/cost tracking) |
| `voice.py`, `logger_setup.py` | kept (speech cleaning now says "2,499 rupees") |

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env        # (Linux/Mac: cp)  then set GROQ_API_KEY, and SERP_API_KEY for web search
python ingest.py              # embeds data/products.json (40 sample products)
```

## Run

```bash
# terminal 1 - tools + search (keep running)
python mcp_server.py

# terminal 2 - the voice bot
python voice_chat.py          # type a request, or press Enter to speak (Enter again to stop)

# terminal 2 (alternative) - web UI + HTTP API, then open http://localhost:8000
uvicorn main:app --port 8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"best earbuds under 3000\"}"
```

Things to say: "cheap wireless earbuds with noise cancelling" / "gaming laptop under 20000" (nothing fits, so it
offers the closest) / "compare the first two" / "add the second one" / "what's in my cart?" / "remove everything".

Debug retrieval without any LLM or voice:
`python query.py "air fryer" --max-price 6000`

## Web UI

`python -m uvicorn main:app --port 8000` serves a chatbot-style front end at http://localhost:8000 (code in `web/index.html`, no build step).

- One conversation, like any chat app. Type, or click the mic, speak, and click again to send (Whisper via `POST /transcribe`; needs `GROQ_API_KEY`).
- Products the assistant finds appear as cards **inside the chat** (swipe sideways), each with an Add to cart button.
- The **Cart** button (top right) opens a drawer: change quantity, remove, see the total. Adding by voice updates it immediately.
- Answers are read aloud by the browser's own voice; the **Voice on/off** button mutes them.
- The microphone needs `localhost` or HTTPS.

Endpoints it uses: `POST /ask` (returns `answer`, `products`, `cart_changed`), `POST /transcribe`, `GET /cart/items`, `POST /cart/add`, `POST /cart/remove`.

## The MCP tools

| Tool | Purpose |
|---|---|
| `search_products` | RAG search. Filters: `max_price`, `min_price`, `category`, `brand`, `min_rating`, `in_stock_only`. Returns `NO_MATCH` / `NO_MATCH_IN_BUDGET` honestly |
| `get_product_details` | Full specs, MRP, discount, stock (accepts an id or a name) |
| `compare_products` | 2-4 products side by side, with cheapest / highest rated |
| `add_to_cart`, `view_cart`, `remove_from_cart` | Persistent cart in `data/cart.json`, stock and quantity checks |
| `search_web` | SerpAPI, for outside reviews, products you don't stock, market prices |

## Using your own catalogue

Replace `data/products.json` (or point `CATALOG_FILE` in `config.py` at a `.csv`), then run `python ingest.py`.

Required fields: `id`, `name`, `category`, `price`. Optional: `brand`, `mrp`, `rating`, `reviews`, `stock`,
`description`, `specs`, `tags`, `image`.

CSV: `specs` as `RAM: 8GB; Storage: 128GB`, `tags` as `wireless|bluetooth`.
If `image` points to a file (relative to `data/`), it is also embedded with CLIP.

Re-run `python ingest.py` after any catalogue change. It only embeds what changed and removes deleted
products. Price and stock shown to the customer are always read from the catalogue file, and the
search re-checks them there, so a stale index can't show a wrong price.

## Tuning (`config.py`)

- `SEARCH_MIN_SCORE`, `RELATIVE_SCORE`: how strict "relevant" is. If good products are missing, lower
  them. If unrelated ones appear, raise them. Check scores with `python query.py`.
- `SYSTEM_PROMPT`: the assistant's behaviour and voice style.
- `REPLY_LANGUAGE` and `STT_LANGUAGE`: e.g. `"Hinglish"` and `""` (auto-detect).
- `CURRENCY`: shown before prices (spoken as "rupees").

## Timing

Per-step timing for each turn is in `logs/pipeline.log` (`Voice answer timing`).

## Test

`python test_smoke.py` runs the whole stack (ChromaDB, MCP server + client, FastAPI, voice loop) offline with a
fake embedder and a scripted LLM, so it needs no downloads or keys.

## Limits to know about

- One shared cart for the whole store (`data/cart.json`). For real users, key it by user/session.
- No checkout or payment. `add_to_cart` is deliberately the last step.
- Search quality depends on the text in `name`, `description`, `specs` and `tags`: put the words customers say there.
