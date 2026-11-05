"""Groq client shared by the agent and the speech-to-text code."""
import os

from groq import Groq

import config  # noqa: F401  (loads .env)
from logger_setup import get_logger

logger = get_logger()

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Put it in .env (see .env.example).")
        _client = Groq(api_key=api_key)
    return _client
