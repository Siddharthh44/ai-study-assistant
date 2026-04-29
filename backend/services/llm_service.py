import os
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logger = logging.getLogger(__name__)
_client_lock = threading.Lock()
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client

    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is missing. Set it in backend/.env.")
            _client = genai.Client(api_key=GEMINI_API_KEY)

    return _client


def _extract_token_usage(response: Any) -> dict[str, int] | None:
    usage_metadata = getattr(response, "usage_metadata", None)
    if usage_metadata is None:
        return None

    usage = {
        "prompt_token_count": getattr(usage_metadata, "prompt_token_count", None),
        "candidates_token_count": getattr(usage_metadata, "candidates_token_count", None),
        "total_token_count": getattr(usage_metadata, "total_token_count", None),
    }

    filtered_usage = {key: value for key, value in usage.items() if isinstance(value, int)}
    return filtered_usage or None


def generate_content(prompt: str, request_size: int | None = None) -> str:
    """
    Sends a prompt to Gemini and returns the response text.
    """
    client = _get_client()
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    except Exception:
        logger.exception(
            "Gemini API call failed timestamp=%s request_size=%s",
            timestamp,
            request_size if request_size is not None else len(prompt),
        )
        raise

    token_usage = _extract_token_usage(response)
    logger.info(
        "Gemini API call timestamp=%s request_size=%s token_usage=%s",
        timestamp,
        request_size if request_size is not None else len(prompt),
        token_usage if token_usage is not None else "unavailable",
    )

    return response.text
