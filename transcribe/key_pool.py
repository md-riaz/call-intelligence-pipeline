"""
Gemini API key pool with automatic rotation on 429 RESOURCE_EXHAUSTED.

Supports multiple keys from .env / config.env so you can pool free-tier
quotas across several Google AI Studio projects:

    # Option A — comma-separated list (easiest to paste)
    GOOGLE_API_KEYS=AIza...key1,AIza...key2,AIza...key3

    # Option B — numbered keys (easy to comment one out)
    GOOGLE_API_KEY_1=AIza...key1
    GOOGLE_API_KEY_2=AIza...key2

    # Option C — original single-key format (still works)
    GOOGLE_API_KEY=AIza...key

3 free-tier keys × 20 RPD = 60 transcriptions/day at no cost.

Thread-safety: not thread-safe. The pipeline is single-threaded.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger(__name__)


class AllKeysExhaustedError(Exception):
    """Raised when every key in the pool has hit RESOURCE_EXHAUSTED."""


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA_EXCEEDED" in msg


class GeminiKeyPool:
    """Round-robin key pool that rotates to the next key on 429."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError(
                "No Google API keys provided. Set GOOGLE_API_KEY (or "
                "GOOGLE_API_KEYS / GOOGLE_API_KEY_1..N for multiple keys) "
                "in your .env or config.env file. "
                "Free keys: https://aistudio.google.com"
            )
        # Deduplicate while preserving order
        self._keys: list[str] = list(dict.fromkeys(k for k in keys if k))
        self._current: int = 0
        self._exhausted: set[int] = set()
        self._clients: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, explicit_key: Optional[str] = None) -> "GeminiKeyPool":
        """Build a pool from: CLI flag > config.env > os.environ."""
        from .config import load_config
        cfg = load_config()

        if explicit_key:
            return cls([explicit_key])

        # Option A: GOOGLE_API_KEYS=key1,key2,...
        raw = cfg.get("GOOGLE_API_KEYS") or os.environ.get("GOOGLE_API_KEYS", "")
        if raw:
            keys = [k.strip() for k in raw.split(",") if k.strip()]
            if keys:
                log.info("Loaded %d API key(s) from GOOGLE_API_KEYS", len(keys))
                return cls(keys)

        # Option B: GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, ...
        keys = []
        i = 1
        while True:
            k = cfg.get(f"GOOGLE_API_KEY_{i}") or os.environ.get(f"GOOGLE_API_KEY_{i}", "")
            if not k:
                break
            keys.append(k)
            i += 1
        if keys:
            log.info("Loaded %d API key(s) from GOOGLE_API_KEY_1..%d", len(keys), i - 1)
            return cls(keys)

        # Option C: single GOOGLE_API_KEY
        key = cfg.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")
        if key:
            return cls([key])

        raise ValueError(
            "No Google API key found. Set GOOGLE_API_KEY in your environment "
            "or config.env file. Get a free key at https://aistudio.google.com"
        )

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def get_client(self) -> tuple[Any, int]:
        """Return (genai.Client, key_index) for the next active key.

        Raises AllKeysExhaustedError if every key is rate-limited.
        """
        if len(self._exhausted) >= len(self._keys):
            raise AllKeysExhaustedError(
                f"All {len(self._keys)} API key(s) have hit the daily rate limit "
                "(RESOURCE_EXHAUSTED). Wait until tomorrow or add more keys — see "
                "GOOGLE_API_KEYS in config.example.env."
            )

        # Advance past any exhausted keys
        for _ in range(len(self._keys)):
            if self._current not in self._exhausted:
                break
            self._current = (self._current + 1) % len(self._keys)

        idx = self._current
        if idx not in self._clients:
            from google import genai
            self._clients[idx] = genai.Client(api_key=self._keys[idx])
            log.debug("Gemini client created for key #%d", idx + 1)

        return self._clients[idx], idx

    def mark_exhausted(self, key_index: int) -> None:
        """Record that key_index returned 429; advance to the next key."""
        self._exhausted.add(key_index)
        remaining = len(self._keys) - len(self._exhausted)
        if remaining > 0:
            log.warning(
                "Key #%d hit RESOURCE_EXHAUSTED — rotating to next key (%d remaining).",
                key_index + 1, remaining,
            )
        self._current = (key_index + 1) % len(self._keys)

    def reset(self) -> None:
        """Clear exhausted set. Call at the start of each day's batch run."""
        self._exhausted.clear()
        self._current = 0

    @property
    def active_count(self) -> int:
        return len(self._keys) - len(self._exhausted)

    @property
    def total_count(self) -> int:
        return len(self._keys)
