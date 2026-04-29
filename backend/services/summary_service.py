import copy
import hashlib
import logging
import os
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from .llm_service import generate_content
from .parser import parse_summary
from .prompt_builder import build_summary_prompt


logger = logging.getLogger(__name__)


def _get_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r. Falling back to %s.", name, value, default)
        return default


CACHE_ENABLED = _get_bool_env("SUMMARY_CACHE_ENABLED", True)
CACHE_TTL_SECONDS = max(_get_int_env("SUMMARY_CACHE_TTL_SECONDS", 600), 0)
DAILY_API_LIMIT = max(_get_int_env("SUMMARY_DAILY_API_LIMIT", 20), 0)
DAILY_API_LIMIT_BUFFER = max(_get_int_env("SUMMARY_DAILY_API_LIMIT_BUFFER", 2), 0)


@dataclass
class CacheEntry:
    value: dict[str, Any]
    expires_at: float


@dataclass
class DailyUsageState:
    current_day: date
    api_calls_started: int = 0


_state_lock = threading.Lock()
_cache: dict[str, CacheEntry] = {}
_in_progress_requests: dict[str, Future] = {}
_daily_usage = DailyUsageState(current_day=datetime.now(timezone.utc).date())


def _make_cache_key(text: str, mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(mode.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _is_cacheable_response(result: Any) -> bool:
    if not isinstance(result, dict):
        return False

    if result.get("error"):
        return False

    summary = result.get("summary")
    key_points = result.get("key_points")

    return isinstance(summary, str) and isinstance(key_points, list)


def _get_cached_result_locked(cache_key: str, now: float) -> dict[str, Any] | None:
    if not CACHE_ENABLED:
        return None

    entry = _cache.get(cache_key)
    if entry is None:
        return None

    if entry.expires_at <= now:
        _cache.pop(cache_key, None)
        return None

    return copy.deepcopy(entry.value)


def _store_cached_result(cache_key: str, result: dict[str, Any], *, cache_allowed: bool = True) -> None:
    if not cache_allowed or not CACHE_ENABLED or CACHE_TTL_SECONDS <= 0 or not _is_cacheable_response(result):
        return

    with _state_lock:
        _cache[cache_key] = CacheEntry(
            value=copy.deepcopy(result),
            expires_at=time.time() + CACHE_TTL_SECONDS,
        )


def _get_request_future(cache_key: str) -> tuple[dict[str, Any] | None, Future | None, bool]:
    now = time.time()

    with _state_lock:
        cached = _get_cached_result_locked(cache_key, now)
        if cached is not None:
            return cached, None, False

        existing_future = _in_progress_requests.get(cache_key)
        if existing_future is not None:
            return None, existing_future, False

        future: Future = Future()
        _in_progress_requests[cache_key] = future
        return None, future, True


def _release_request_future(cache_key: str) -> None:
    with _state_lock:
        _in_progress_requests.pop(cache_key, None)


def _reserve_daily_api_call_slot() -> bool:
    if DAILY_API_LIMIT <= 0:
        return True

    today = datetime.now(timezone.utc).date()
    near_limit_threshold = max(DAILY_API_LIMIT - DAILY_API_LIMIT_BUFFER, 0)

    with _state_lock:
        if _daily_usage.current_day != today:
            _daily_usage.current_day = today
            _daily_usage.api_calls_started = 0

        if _daily_usage.api_calls_started >= near_limit_threshold:
            return False

        _daily_usage.api_calls_started += 1
        return True


def _build_limit_response() -> dict[str, Any]:
    return {
        "title": "Usage limit reached",
        "summary": "Daily summarization capacity is nearly exhausted. Please try again tomorrow.",
        "key_points": [
            "Summary generation is temporarily paused to avoid exceeding the daily Gemini API limit."
        ],
        "important_terms": [],
    }


def _call_summary_pipeline(text: str, mode: str) -> tuple[dict[str, Any], bool]:
    if not _reserve_daily_api_call_slot():
        logger.warning("Daily Gemini safeguard triggered for request_size=%s mode=%s", len(text), mode)
        return _build_limit_response(), False

    prompt = build_summary_prompt(text, mode)
    llm_response = generate_content(prompt, request_size=len(text))
    return parse_summary(llm_response), True


def generate_summary(text: str, mode: str = "short") -> dict[str, Any]:
    """
    Full pipeline for generating a summary, with caching and in-flight deduplication.
    """

    cache_key = _make_cache_key(text, mode)
    cached_result, future, is_owner = _get_request_future(cache_key)

    if cached_result is not None:
        return cached_result

    assert future is not None

    if not is_owner:
        return copy.deepcopy(future.result())

    try:
        result, cache_allowed = _call_summary_pipeline(text, mode)
        _store_cached_result(cache_key, result, cache_allowed=cache_allowed)
        future.set_result(copy.deepcopy(result))
        return result
    except Exception as exc:
        future.set_exception(exc)
        raise
    finally:
        _release_request_future(cache_key)
