"""In-process cost-guard rate limiting (see docs/03-cors-and-rate-limiting.md).

Not a real per-user rate limiter -- there is exactly one legitimate caller (the
backend, authenticated by the single shared API key from app.api.auth), so this can
only ever see "the backend" as a caller, not individual end users. This exists purely
to cap the worst case: a malfunctioning retry loop or a leaked key running up the GCP
compute / LLM bill before a human notices. Deliberately blunt: a fixed-window counter,
no new dependency, tuned generously above realistic peak legitimate traffic.
"""

from __future__ import annotations

import os
import time


def _int_env(name: str, fallback: int) -> int:
    """Read an int env var, falling back to a hardcoded default if unset or not a
    valid int -- a typo'd override should degrade to the safe default, not crash the
    service or silently disable the rate limit (e.g. an empty string or "0" mistake).
    """
    raw = os.getenv(name)
    if not raw:
        return fallback
    try:
        value = int(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


# Both are overridable per-deploy (docs/03-cors-and-rate-limiting.md §2.2), but the
# hardcoded values here are the actual safety net -- any unset/invalid env var falls
# back to them rather than failing open with no limit at all.
WINDOW_SECONDS = _int_env("RATE_LIMIT_WINDOW_SECONDS", 60)

# Generous default: batch endpoints (e.g. /forecast/daily-batch) exist precisely so a
# real integration never needs to call per-item in a loop, so legitimate per-minute
# volume should stay well under this even at peak.
DEFAULT_LIMIT_PER_MIN = _int_env("RATE_LIMIT_DEFAULT_PER_MIN", 300)

# (caller, tier) -> (window_start_monotonic, count). One process, one event loop:
# plain dict mutation is safe without locking since nothing awaits between read and
# write here.
_COUNTERS: dict[tuple[str, str], tuple[float, int]] = {}


def _tier(path: str) -> tuple[str, int]:
    """Which bucket a route falls into, and that bucket's limit."""
    return "default", DEFAULT_LIMIT_PER_MIN


def limit_for(path: str) -> int:
    """The requests-per-window ceiling that applies to a given route path."""
    return _tier(path)[1]


def check(caller_id: str, path: str, *, window_seconds: int = WINDOW_SECONDS) -> tuple[bool, int]:
    """Increment the counter for this caller+tier, resetting it if its window elapsed.

    Returns (allowed, retry_after_seconds). `retry_after_seconds` is only meaningful
    when `allowed` is False -- seconds remaining until the window resets.
    """
    tier, limit = _tier(path)
    key = (caller_id, tier)
    now = time.monotonic()
    window_start, count = _COUNTERS.get(key, (now, 0))
    if now - window_start >= window_seconds:
        window_start, count = now, 0
    count += 1
    _COUNTERS[key] = (window_start, count)
    if count > limit:
        retry_after = max(1, int(window_seconds - (now - window_start)))
        return False, retry_after
    return True, 0
