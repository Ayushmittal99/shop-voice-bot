"""Local embedding models. Loaded lazily so import is cheap."""
from sentence_transformers import SentenceTransformer

from config import IMAGE_MODEL, TEXT_MODEL
from logger_setup import get_logger

logger = get_logger()

_text_model = None
_image_model = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        logger.info(f"Loading text model: {TEXT_MODEL}")
        _text_model = SentenceTransformer(TEXT_MODEL)
    return _text_model


def _get_image_model():
    global _image_model
    if _image_model is None:
        logger.info(f"Loading image model: {IMAGE_MODEL}")
        _image_model = SentenceTransformer(IMAGE_MODEL)
    return _image_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_text_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_images(paths: list[str]) -> list[list[float]]:
    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in paths]
    return _get_image_model().encode(
        images, batch_size=16, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def embed_query_for_images(query: str) -> list[float]:
    """CLIP embeds text into the same space as images, so text queries
    can retrieve product photos directly."""
    return _get_image_model().encode([query], normalize_embeddings=True).tolist()[0]
