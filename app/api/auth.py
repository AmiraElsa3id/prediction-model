"""API key auth for the whole service (see docs/01-api-key-hardening.md).

Replaces the old, RestoMind-only `AI_SHARED_SECRET` / `X-RestoMind-Key` guard: every
route except `/health` now requires `X-API-Key`, and the deploy-time secret is the
SHA-256 hash of the real key (`API_KEY_HASH`), not the raw value -- the raw key is never
an env var on this service (docs §2.2). `REQUIRE_API_KEY` controls whether missing config
fails closed (production) or is skipped (local dev), matching docs §2.4.
"""

from __future__ import annotations

import hashlib
import hmac
import os

# Read once at import time so a misconfigured deploy fails loudly at startup instead of
# quietly serving unauthenticated traffic at the first request (docs §2.4 / §4.1).
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"
API_KEY_HASH = os.getenv("API_KEY_HASH")

if REQUIRE_API_KEY and not API_KEY_HASH:
    raise RuntimeError(
        "REQUIRE_API_KEY is true but API_KEY_HASH is unset. Generate a key and its hash "
        "(see docs/01-api-key-hardening.md §3.2) and set API_KEY_HASH before starting "
        "this service -- refusing to boot into a silently-open state."
    )

# Routes reachable with no key at all: load balancers / orchestrators / uptime checks
# need to hit /health without a credential (docs §2.3).
EXEMPT_PATHS = frozenset({"/health"})


def is_valid_key(presented_key: str) -> bool:
    """Constant-time check of a raw `X-API-Key` header value against `API_KEY_HASH`."""
    if not API_KEY_HASH:
        return False
    digest = hashlib.sha256(presented_key.encode()).hexdigest()
    return hmac.compare_digest(digest, API_KEY_HASH)

