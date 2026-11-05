"""ChromaDB wrapper with two collections: products (text) and product_images."""
import chromadb

from config import CHROMA_DIR, IMAGE_COLLECTION, PRODUCT_COLLECTION
from logger_setup import get_logger, log_event

logger = get_logger()


class VectorStore:
    def __init__(self):
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # cosine distance matches the normalized embeddings
        self.products = self.client.get_or_create_collection(
            PRODUCT_COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        self.images = self.client.get_or_create_collection(
            IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.products.count()

    def stored_hashes(self) -> dict[str, str]:
        """{product_id: content_hash} of everything currently indexed (used to ingest only changes)."""
        res = self.products.get(include=["metadatas"])
        return {i: (m or {}).get("content_hash", "") for i, m in zip(res["ids"], res["metadatas"])}

    def delete(self, ids: list[str]):
        if ids:
            self.products.delete(ids=ids)
            self.images.delete(ids=ids)
            log_event(logger, "Deleted products from index", count=len(ids))

    def add_products(self, ids, embeddings, documents, metadatas):
        self.products.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def add_images(self, ids, embeddings, documents, metadatas):
        self.images.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def search(self, query_embedding, k: int = 5, where: dict | None = None):
        n = min(k, max(self.products.count(), 1))
        return self.products.query(query_embeddings=[query_embedding], n_results=n, where=where)

    def search_images(self, query_embedding, k: int = 3, where: dict | None = None):
        n = min(k, max(self.images.count(), 1))
        return self.images.query(query_embeddings=[query_embedding], n_results=n, where=where)
