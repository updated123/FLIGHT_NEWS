"""Groq API helpers with rate-limit retry."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Optional

from groq import Groq

DEFAULT_MAX_RETRIES = 5
DEFAULT_DELAY_BETWEEN_CALLS = float(os.getenv("GROQ_REQUEST_DELAY_SECONDS", "7"))


def parse_retry_seconds(error_message: str) -> float:
    match = re.search(r"try again in ([\d.]+)s", error_message, re.IGNORECASE)
    if match:
        return float(match.group(1)) + 1.0
    return DEFAULT_DELAY_BETWEEN_CALLS


def groq_chat_with_retry(
    client: Groq,
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    **kwargs: Any,
):
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_error = exc
            message = str(exc)
            if "429" not in message and "rate_limit" not in message.lower():
                raise
            wait = parse_retry_seconds(message)
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                raise
    if last_error:
        raise last_error
    raise RuntimeError("Groq request failed after retries")


def throttle_between_calls() -> None:
    time.sleep(DEFAULT_DELAY_BETWEEN_CALLS)
