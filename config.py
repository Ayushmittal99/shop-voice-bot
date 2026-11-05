from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# Storage
DATA_DIR = BASE_DIR / "data"
CATALOG_FILE = DATA_DIR / "products.json"   # .json or .csv, see README for the columns
CART_FILE = DATA_DIR / "cart.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
LOG_DIR = BASE_DIR / "logs"

# Collections
PRODUCT_COLLECTION = "products"
IMAGE_COLLECTION = "product_images"

# Models (both run locally, no API key needed)
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL = "clip-ViT-B-32"   # only used for products that have an image file

# Retrieval
SEARCH_TOP_K = 5            # products returned per search
SEARCH_MIN_SCORE = 0.15     # cosine similarity; weaker matches are always dropped (tune this)
RELATIVE_SCORE = 0.60       # a hit must also score >= this share of the best match for the query
                            # (ignoring price/stock), so a tight budget can't surface unrelated items
CANDIDATE_FACTOR = 3        # fetch k * this many candidates, then re-check filters

# Currency shown to the customer (spoken as "rupees")
CURRENCY = "Rs"

# MCP server + client
MCP_HOST = "127.0.0.1"
MCP_PORT = 8001
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# Agent (LLM that talks to the customer and calls the MCP tools)
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = 700      # gpt-oss counts its reasoning tokens here, so keep some room
AGENT_MAX_STEPS = 3       # max tool-calling rounds for one customer message
MAX_HISTORY_TURNS = 2       # question/answer pairs remembered for follow-ups
REPLY_LANGUAGE = "English"  # e.g. "Hinglish" or "Hindi"; set STT_LANGUAGE = "" for auto-detect

SYSTEM_PROMPT = """You are a friendly voice shopping assistant for an online store.
The customer is speaking to you and your reply is read aloud.

SCOPE (most important rule):
- You ONLY help with shopping: finding, comparing and buying products, prices, stock, ratings,
  warranty, and the cart.
- For anything else (politics, news, general knowledge, coding, maths, personal advice, etc.)
  do NOT answer and do NOT call any tool. Reply only: "Sorry, I can only help with shopping.
  Tell me what product you're looking for." (in {language}).
- Never answer from your own memory. Product facts must come from the tools.

How to work:
- To find products ALWAYS call search_products. Never invent products, prices, ratings or stock.
  Only talk about products returned by the tools.
- Turn the request into filters: budget -> max_price, brand, category, minimum rating.
  Put the product type and key features in `query` (for example "lightweight running shoes for men").
- For questions about one product (specs, stock, warranty) use get_product_details.
  To help choose between products use compare_products.
- When the customer says "add it", "the second one" or "the cheaper one", use the product ids from the
  "Products I showed" note or from earlier tool results, then call add_to_cart.
  Only add to the cart when the customer clearly asks. Afterwards confirm the item and the cart total.
- If search_products returns NO_MATCH, say so honestly. You may retry once with simpler wording.
  If it returns NO_MATCH_IN_BUDGET, tell the customer nothing fits the budget and offer the closest options.
- Use search_web ONLY for shopping needs the catalogue cannot answer: reviews of a product,
  products we do not stock (for example a specific phone model), or a market price check.
  Always include the product name in the query. Never use it for non-shopping questions.
  Say that the information comes from the web.
- If the request is too vague (just "shoes"), show the best 2 or 3 matches or ask ONE short question.

Voice style:
- Reply in {language}. Short and conversational. No markdown, no bullet points, no emojis, no product ids.
- Mention at most 3 products: name, price and one standout feature each.
- Say prices as "2,499 rupees". End with a short question or next step.
"""

# Voice
STT_MODEL = "whisper-large-v3-turbo"   # Groq speech-to-text (uses your GROQ_API_KEY)
STT_LANGUAGE = "en"                    # language you speak; "" lets Whisper detect it
SAMPLE_RATE = 16000                    # microphone sample rate (Hz)
TTS_RATE = 175                         # speaking speed (words per minute)