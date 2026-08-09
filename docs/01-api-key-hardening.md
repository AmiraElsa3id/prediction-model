# 01 — API key hardening for all endpoints

**Status:** Implemented (`app/api/auth.py`, `app/api/main.py`'s `_require_api_key`
middleware, `tests/test_api.py`, `.env.example`, `README.md`/`HANDOFF.md`,
`postman_collection.json`). Rate limiting, per-tenant credentials, CORS cleanup, and
rotation automation remain open per §7.
**Owner context:** this service is an internal microservice. The only intended caller is
the backend (RestoMind's NestJS API today, possibly others later). It is not meant to be
reachable from a browser or the public internet directly. This plan makes that assumption
enforced in code, not just true by accident of network topology.
**Scope:** authentication only. Rate limiting, CORS, and payload-size limits are explicitly
OUT of scope for this doc — tracked separately, ignore them while implementing this.

---

## 1. Why this is needed

Today (`app/api/main.py`), exactly one thing is authenticated: the RestoMind integration
routes (`/integration/restomind/*`), and only if the `AI_SHARED_SECRET` env var happens to
be set — if it's unset, those routes are open. Every other route
(`/forecast/*`, `/data/ingest`, `/marketing/generate-offer`, `/marketing/publish`,
`/surplus/detect`, `/alerts/waste-prevention`) has **no auth at all**, unconditionally.

For a microservice whose only legitimate caller is our own backend, "no auth" means
"anyone who can route a packet to this port has full access" — including triggering a
real Facebook post via `/marketing/publish` if the Meta env vars are ever turned on, or
poisoning another tenant's learned demand levels via the registry endpoints. Network
isolation (private subnet, no public ingress) is the first line of defense, but it
shouldn't be the *only* one — a misconfigured security group, a debug port left open, or
a future deploy that puts this behind a shared ingress are all one mistake away from
exposing an unauthenticated service. An API key is cheap insurance against exactly that
class of mistake.

---

## 2. Design

### 2.1 Auth model

Single shared API key (bearer-token style, not per-tenant, not OAuth/JWT). This is
deliberate, not a shortcut:

- There is exactly one legitimate caller (the backend), so a single credential is enough
  — no need for user-level identity, scopes, or expiry semantics inside this service.
- The backend already holds and manages its own secrets; this just adds one more.
- Per-tenant auth (e.g. a key per `restaurantId`) is a different, bigger feature (real
  tenant isolation) and is explicitly NOT what this plan does. See §7 for that gap.

### 2.2 Where it lives

- Env var: `API_KEY` (see §2.5 for what happens to the existing `AI_SHARED_SECRET`).
- Sent as a header on every request: `X-API-Key: <key>`.
  - Use a dedicated header, not `Authorization: Bearer`, to avoid any ambiguity with
    future OAuth/JWT work and to match the existing `X-RestoMind-Key` convention already
    in this codebase.
- **Decided: stored and compared as a hash, not the raw key.** This mirrors how
  today's model-provider APIs (OpenAI, Anthropic, Stripe, etc.) handle API keys: the raw
  value is shown once at generation time and never persisted anywhere — what's stored is
  a SHA-256 hash of it. Concretely:
  - The env var actually deployed to this service is `API_KEY_HASH` (hex-encoded SHA-256
    digest of the real key), not the raw `API_KEY` value.
  - On each request, hash the incoming `X-API-Key` header value with SHA-256 and compare
    that digest against `API_KEY_HASH` using `hmac.compare_digest` (constant-time, same
    property the existing `_require_shared_secret` middleware already relies on — reuse
    that comparison call, don't reinvent it).
  - This means even if the deploy environment's config/env-var store is read by someone
    who shouldn't have access, they get a hash, not a usable key.
  - The raw key itself only ever needs to live on the backend side (the caller) and
    briefly in whoever's terminal/secrets-manager generates it — see §3.3.

### 2.3 What's protected

**Everything except `/health`.**

- `/health` stays open, unauthenticated. Load balancers, container orchestrators, and
  uptime checks need to hit it without a credential, and it leaks nothing sensitive
  (`status`, whether the model is trained, item count, and the fact that data is
  simulated — see `app/api/main.py:136-144`).
- Every other route — all of `/forecast/*`, `/data/ingest`, `/model/status`,
  `/alerts/waste-prevention`, `/surplus/detect`, `/marketing/*`, and all of
  `/integration/restomind/*` — requires the key.
- `/docs`, `/redoc`, `/openapi.json` (FastAPI's auto-generated Swagger UI): **decide
  before implementing** whether these stay open or require the key too. Recommendation:
  require the key in any environment where `REQUIRE_API_KEY` is enforced (see §2.4) —
  Swagger UI exposes the full request/response shape of every route, which is
  reconnaissance value to an attacker even if it can't execute calls. If keeping `/docs`
  open for convenience during integration work, say so explicitly in the env var
  documentation so it's a decision, not an oversight.

### 2.4 Fail closed, not open

This is the most important behavioral change from the existing pattern. Today, an unset
`AI_SHARED_SECRET` silently means "no auth" (`app/api/main.py:82`: `if secret and ...`).
That's fine for local dev but dangerous as a production default — an env var that fails
to get set in a deploy should not silently downgrade to "wide open."

New behavior:

- Add `REQUIRE_API_KEY` (or reuse `ENV`/`ENVIRONMENT` if the deploy already sets one —
  check before inventing a new var). Two modes:
  - **Dev mode** (`REQUIRE_API_KEY=false`, or unset): auth is skipped entirely if
    `API_KEY_HASH` is also unset, so `uvicorn app.api.main:app --reload` keeps working
    with zero config, exactly like today. This is the default for local development only.
  - **Enforced mode** (`REQUIRE_API_KEY=true`): `API_KEY_HASH` MUST be set, and the app
    should refuse to start (raise at import/startup time, not just 401 every request) if
    it isn't. Fail loud at boot, not quietly at the first request someone happens to send.
- Whatever mechanism is chosen, the deploy config (docker-compose / k8s manifest / whatever
  ships this) must set `REQUIRE_API_KEY=true` for any environment other than a developer's
  own machine. Call this out explicitly in the PR/deploy checklist when this is built —
  it's the step most likely to get silently skipped.

### 2.5 Consolidating with the existing `AI_SHARED_SECRET` / `X-RestoMind-Key`

**Decided:** remove `AI_SHARED_SECRET` entirely. It is not being kept alongside the new
scheme, not even temporarily behind a second accepted header — `API_KEY` / `X-API-Key`
becomes the one and only auth mechanism for this service.

- Delete `_require_shared_secret` and its narrower path check (`main.py:74-90`).
- Update `RestoMindAPI`-side callers (and `app/integration/connect_restomind.py`,
  `seed_restomind.py` if they send the old header) to send `X-API-Key` instead of
  `X-RestoMind-Key`, in the same change that removes the old middleware — there is no
  migration window where both are accepted.
- Search the repo for `AI_SHARED_SECRET` and `X-RestoMind-Key` before starting so nothing
  is missed (`app/api/main.py`, any deploy scripts/env templates, `.env.example`,
  `README.md`, `HANDOFF.md`, `LIVE_DEMO.md`, and anything under `app/integration/` that
  constructs outgoing requests to this service).

---

## 3. Key generation

### 3.1 Requirements

- Cryptographically random, not a password a human picks.
- At least 256 bits of entropy (32 random bytes), which comfortably exceeds what
  brute-forcing over a network could ever threaten.
- URL/header-safe encoding (no characters that need escaping in an HTTP header value).

### 3.2 How to generate one

Either of these produces a suitable key — use whichever is convenient in the deploy
tooling:

```bash
# Option A: OpenSSL (widely available, good for shell scripts / CI)
openssl rand -hex 32
# -> 64 hex characters, 256 bits of entropy

# Option B: Python stdlib (no extra dependency, matches the codebase's language)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
# -> ~43 URL-safe base64 characters, 256 bits of entropy
```

Then derive the `API_KEY_HASH` that actually gets deployed (§2.2) from the raw value:

```bash
python3 -c "import hashlib,sys; print(hashlib.sha256(sys.argv[1].encode()).hexdigest())" '<raw key>'
```

Give the raw key to the backend (§3.3) and the deploy pipeline the hash — never store the
raw value in this service's own environment.

Do not:
- Use `uuid4()` — UUIDs are not designed as secrets and have less usable entropy than a
  raw random token of the same length.
- Derive the key from anything guessable (project name, date, restaurant ID).
- Commit a generated key anywhere in the repo, including in test fixtures beyond a
  clearly-fake placeholder (e.g. `test-key-not-real`) used only under `pytest`.

### 3.3 Storage

- **Backend side (the caller):** wherever the backend already keeps its other secrets
  (DB credentials, Meta tokens, etc.) — a secrets manager or environment-injected config,
  never a committed file. This key is exactly as sensitive as those.
- **This service side:** `API_KEY` environment variable, injected by whatever deploys this
  service (same mechanism as `META_ACCESS_TOKEN` today). Never written to
  `data/registry.json` or any log line — audit the code for accidental logging of request
  headers before shipping this (FastAPI/Starlette don't log headers by default, but
  double-check any custom logging middleware added later doesn't start).
- **Local dev:** an untracked `.env` file (confirm `.env` is in `.gitignore` — check before
  relying on this) or exported in the shell. Never needed at all if `REQUIRE_API_KEY` is
  left at its dev default (§2.4).

### 3.4 Rotation

**Decided: manual only, for now.** No automated/scheduled rotation is in scope for this
plan — that's explicitly deferred (see §7).

- Rotating means: generate a new key + hash it (§3.2), update it on both sides (this
  service's `API_KEY_HASH` env var and the backend's stored raw secret), redeploy this
  service first so it accepts the new key, then update the backend to send it. A brief
  overlap where the service accepts only the new key while the backend still sends the
  old one will 401 — plan the rotation as a short manual deploy window.
- If rotation becomes frequent enough that this manual window is a problem, revisit
  §2.5's "accept two headers" idea and extend it to "accept two valid key hashes during a
  rotation window" — but that's future work, not part of this plan.
- No fixed rotation schedule is being mandated by this plan. Rotation happens only when
  someone manually triggers it (e.g. suspected leak, offboarding), not on a timer.

---

## 4. Implementation sketch

This section is a guide for whoever builds this, not a diff to paste — the current code
will have moved on by the time this is picked up.

1. Add a small auth helper (could live in `app/api/main.py` next to the existing
   `_require_shared_secret`, or a new `app/api/auth.py` if `main.py` is getting crowded):
   - Reads `API_KEY_HASH` and `REQUIRE_API_KEY` at import time (module-level, like the
     existing pattern reads `AI_SHARED_SECRET` inside the middleware — either works, but
     reading once at startup makes the "refuse to boot" behavior in §2.4 easier to
     implement than reading per-request).
   - Raises at import/startup if `REQUIRE_API_KEY` is true and `API_KEY_HASH` is unset or
     empty.
2. Register it as a `@app.middleware("http")` function, same shape as
   `_require_shared_secret` today, but:
   - Checks `request.url.path` against an exemption list (`{"/health"}` plus optionally
     `/docs`, `/openapi.json`, `/redoc` per the §2.3 decision) instead of a path prefix.
   - Hashes `request.headers.get("X-API-Key", "")` with SHA-256 and compares the resulting
     hex digest against `API_KEY_HASH` via `hmac.compare_digest` (§2.2) — never compares
     the raw header value directly against anything stored.
   - Returns the same `ErrorResponse` shape already used elsewhere
     (`app/api/schemas.py`'s `ErrorResponse`) on failure, status 401, so error handling
     stays consistent across the API rather than introducing a new shape.
3. Keep the middleware registration order comment that's already in the code
   (`app/api/main.py:96-102`) accurate — if CORS is ever reintroduced or changed,
   whichever middleware needs to run outermost (to handle OPTIONS preflights or attach
   headers to error responses) must be registered last. Re-verify this ordering
   requirement holds for whatever the final CORS decision is (tracked separately, but
   don't break it as a side effect of this change).
4. Delete `_require_shared_secret` and the `AI_SHARED_SECRET`/`X-RestoMind-Key` names once
   the replacement is confirmed working end-to-end (§2.5).

---

## 5. Testing

- Extend `tests/test_api.py` (or wherever the FastAPI `TestClient` fixtures live) with:
  - A request to a protected route with no key → 401.
  - A request with a wrong key → 401.
  - A request with the correct key → 200 (existing behavior unchanged).
  - A request to `/health` with no key → 200 (stays open).
  - If `/docs`/`/openapi.json` are protected per §2.3, a test for that too.
- The existing test suite presumably runs with `REQUIRE_API_KEY` unset/false (dev mode) so
  it keeps working without every single test needing to inject a header — confirm this is
  still true after the change, and add one test module/fixture that explicitly sets
  `REQUIRE_API_KEY=true` + `API_KEY=<test value>` to exercise the enforced path, mirroring
  how `REGISTRY_STORE=""` is already used to isolate the test suite from persistence
  (`app/api/main.py:53-57`).
- Update `postman_collection.json` to send `X-API-Key` (as a collection-level header/env
  var) so manual/newman testing doesn't silently start failing once this ships.

---

## 6. Docs to update once this is built

- `README.md`'s env var table (currently documents `CORS_ORIGINS`,
  `META_PAGE_ID`/`META_ACCESS_TOKEN`/`META_PUBLISH_ENABLED`, etc.) — add `API_KEY_HASH`
  and `REQUIRE_API_KEY` (and note that the raw key itself is never an env var on this
  service — see §2.2), and remove `AI_SHARED_SECRET` per §2.5 (removed, not deprecated).
- `HANDOFF.md` §2 (env vars list) and §6 (endpoint list) — note that all routes except
  `/health` now require `X-API-Key`, and that rotation is manual-only (§3.4).
- Whatever deploy manifest/compose file exists — make sure `API_KEY_HASH` is documented as
  the required secret injection (derived per §3.2, raw key never stored here), and
  `REQUIRE_API_KEY=true` is set for any non-local environment.

---

## 7. Explicitly out of scope / open questions for later

- **Rate limiting** — separate plan, not this one.
- **Per-tenant credentials** for the RestoMind bridge (today's single shared key means any
  holder can act as any `restaurantId` — real tenant isolation would need a key or token
  per restaurant, checked against the `restaurantId` in the request body). Worth a future
  plan once/if this service handles multiple untrusted backend tenants directly rather
  than one trusted backend that itself handles multi-tenancy.
- **CORS** — if the decision from the earlier discussion (browser never calls this service
  directly, only the backend does) is confirmed, the `CORSMiddleware` block in
  `app/api/main.py:103-108` becomes dead code and should be removed in a follow-up. Not
  part of this plan; flagging so it isn't forgotten.
- **Key rotation automation / schedule** — no schedule mandated here (§3.4); decide based
  on actual deploy environment ownership.
- **`/docs` exposure** — needs an explicit decision (§2.3) before implementation, not left
  to whoever's typing at the time.
