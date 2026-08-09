# 03 — CORS lockdown and cost-guard rate limiting

**Status:** Implemented. CORS removed entirely (option A, §2.1 — confirmed: this
service's owner context already states no browser calls it directly). Rate limiting
built per §2.2/§3, with one deliberate deviation from the original sketch: the two
tier limits and window are env-overridable (`RATE_LIMIT_DEFAULT_PER_MIN`,
`RATE_LIMIT_MARKETING_PER_MIN`, `RATE_LIMIT_WINDOW_SECONDS`, all falling back to the
hardcoded 300/20/60 defaults below if unset or invalid), not fixed code constants —
requested during implementation so the ceiling can be tuned per-deploy without a code
change once real traffic is observed (see §2.2's original "don't treat these as final"
note, which this satisfies more directly).
**Owner context:** same as `docs/01-api-key-hardening.md` — this service is an internal
microservice. Its one legitimate caller is the backend (RestoMind's NestJS API today,
acting **on behalf of many end users**, not calling once per human). That last point
drives most of the design here: a browser never calls this service directly today (see
§2.1), and because the caller is a backend fanning out on behalf of an unknown number of
real users, per-user throttling isn't something this service can do at all — it only
ever sees "the backend," authenticated by one shared API key (`docs/01`). What this
service CAN and SHOULD do is protect itself (and the infra bill) from that one caller's
traffic spiking far beyond anything a real, working integration would ever produce.
**Scope:** CORS configuration + one blunt, cost-focused rate limit. This is explicitly
**not** a fairness/UX rate limiter (no per-user quotas, no 429-as-a-feature) — see §1.2
for why that distinction matters and shapes every decision below. Auth is out of scope
here — see `docs/01-api-key-hardening.md`, already implemented.

---

## 1. Why this is needed

### 1.1 CORS

Today (`app/api/main.py`, `CORSMiddleware` block), the allowed **origins** are already
restricted to `CORS_ORIGINS` (default `http://localhost:3000`) — not `"*"` — but
`allow_methods=["*"]` and `allow_headers=["*"]` are wide open, and the default origin is
a local dev value with no enforced production override. CORS only matters at all if a
**browser** is a legitimate direct caller. Confirm that's true before spending effort
tightening this — if the answer is "no, only the backend ever calls us, browsers go
through the backend," the correct fix is removing `CORSMiddleware` entirely (already
flagged as a possible follow-up in `docs/01-api-key-hardening.md` §7), not narrowing it.
This doc covers both paths — decide which one applies during implementation (§2.1).

### 1.2 Rate limiting — and why this is a cost guard, not real rate limiting

Normal API rate limiting exists to keep one user from starving others, or to enforce a
pricing tier. **Neither applies here.** There is exactly one caller (the backend, per
`docs/01`'s single-shared-key model), so there's no "other user" to protect from a noisy
neighbor, and no tiers to enforce. If this service had a genuine per-user rate limit,
it would need per-user identity, which it deliberately does NOT have (`docs/01` §2.1 /
§7 — per-tenant credentials are explicitly out of scope).

What actually matters: this service does real compute on every request —

- `/forecast/*`, `/surplus/detect`, `/integration/restomind/*` — model inference (CPU),
  cheap per-call but not free at volume.

If the backend has a bug (a retry loop with no backoff, a cron misfire calling this in a
tight loop, a queue replaying the same job thousands of times) or its own API key leaks
and gets abused, this service has **no ceiling today** — it will happily accept and
execute every request, running up the GCP compute bill, as fast as the caller can send
requests. That's the failure mode this plan protects against: **not** "is this fair to
users," but **"can one malfunctioning or compromised caller run up an unbounded bill
before a human notices and intervenes."**

This reframing matters for every design choice below: the limit should be set generously
above realistic peak legitimate traffic (so it never gets in the way of normal
operation).

---

## 2. Design

### 2.1 CORS

**Decide before implementing which of these applies** (ask whoever owns the RestoMind
integration if unclear):

- **(A) No browser ever calls this service directly** — the backend is a proxy/gateway
  and browsers only ever talk to the backend, which talks to this service server-to-server.
  If true: remove `CORSMiddleware` entirely. It becomes dead code that does nothing but
  create a false sense of a browser-facing security boundary that isn't actually needed,
  and removing it also removes the middleware-ordering subtlety documented at
  `app/api/main.py:97-103` (CORS must stay outermost relative to the API-key guard) —
  one less thing to get wrong in future changes.
- **(B) A browser frontend does call this service directly** (e.g. an admin/investor
  dashboard, or `dashboard.py`'s Streamlit app if it ever moves to browser-side fetches
  instead of server-side Python calls) — keep `CORSMiddleware` but tighten it:
  - `allow_origins`: keep reading `CORS_ORIGINS`, but **require it to be set explicitly
    in any non-local environment** — fail startup (same "fail closed" pattern as
    `docs/01` §2.4) if `CORS_ORIGINS` is unset AND `REQUIRE_API_KEY=true` (i.e. a real
    deploy), rather than silently falling back to the `localhost:3000` dev default in
    production.
  - `allow_methods`: replace `["*"]` with the literal list this service actually uses —
    `["GET", "POST", "OPTIONS"]` (check the route table in `docs/01` §2.3 / `HANDOFF.md`
    §6 for the full set; there is no PUT/DELETE/PATCH anywhere today). Wildcarding
    methods costs nothing to lock down and removes one degree of freedom for a
    misconfigured/malicious cross-origin caller.
  - `allow_headers`: replace `["*"]` with `["Content-Type", "X-API-Key"]` — the only two
    headers any real caller needs to send. A wildcard here is meaningless as a security
    boundary anyway (headers aren't secret) but being explicit documents intent and
    matches the spirit of §2.3 below (name every deliberate exception, don't leave one by
    omission).
  - If a browser is calling directly, note that it will need to send `X-API-Key` from
    client-side code, which means the key is not really secret from anyone who opens
    devtools — flag this as a real gap to whoever owns that decision; it may mean the
    long-term fix is "browser calls the backend, backend calls us" (i.e. converge on
    option A) rather than hardening a browser-exposed key further.

### 2.2 Rate limiting — mechanism

**Decision: a single, generous, in-process limiter — no new external dependency, no
Redis.** This service is not deployed as a fleet of instances behind a shared limiter
today (SQLite/pickle-backed `RegistryStore`, in-memory `STATE` per `docs/01` context, one
`uvicorn` process); an in-memory counter that resets per-process is enough to catch a
single caller sending obviously-abnormal volume, and matches this codebase's existing
preference for stdlib-first solutions (`app/api/auth.py` uses `hashlib`/`hmac`, no new
package for the API-key work either). If this service is ever scaled to multiple
instances behind a load balancer, revisit — a per-instance limit divided across N
instances is a weaker guarantee (see §7) — but don't add that complexity before it's
actually needed.

- **Granularity: per caller, not per route-and-caller**, with one exception (§2.3). Since
  there is exactly one legitimate caller (the backend, single shared key), "per caller"
  in practice means "globally, across this whole service" for the common case. Implement
  it keyed by the presented API key (or by source IP as a fallback in dev mode where no
  key is required) so the mechanism is still meaningful if a second caller/key is ever
  introduced later, without being over-engineered for that case now.
- **Algorithm: fixed-window counter**, not token bucket / sliding log. This is a blunt
  cost guard, not a smooth-traffic-shaping tool — simplicity and an easy-to-reason-about
  limit ("N requests per M-second window") beats precision here. A window of **60
  seconds** is a reasonable default to tune from.
- **Limits (starting points — tune against real traffic once observed, don't treat these
  as final):**
  - **Default (cheap routes — forecasting, surplus, restomind bridge, health exempted):**
    generous, e.g. 300 requests/minute. This should be well above anything a legitimately
    integrated backend would send even at peak (batch endpoints like
    `/forecast/daily-batch` exist specifically so the backend does NOT need to call
    per-item in a loop — see `app/api/main.py`'s docstring there — so per-minute call
    volume should stay low even under real load).
- **On exceeding the limit:** `429 Too Many Requests`, using the same `ErrorResponse`
  shape as the API-key guard (`schemas.ErrorResponse`, matching the pattern already
  established in `docs/01` §4 step 2) so error handling stays consistent. Include a
  `Retry-After` header (seconds until the window resets) — cheap to add, and lets a
  well-behaved backend back off correctly instead of hammering the limit repeatedly.

### 2.3 `/health` stays exempt

Same reasoning as the API-key guard (`docs/01` §2.3): load balancers and uptime checks
poll this frequently and it's not a cost-sensitive route. Exempt it from rate limiting
the same way it's exempt from auth — reuse `auth.EXEMPT_PATHS` as the base exemption set
if convenient, or a parallel constant if the rate limiter ships as a fully separate
module (§4).

---

## 3. Implementation sketch

This section is a guide for whoever builds this, not a diff to paste — the current code
will have moved on by the time this is picked up.

1. **CORS** (§2.1): resolve option A vs B first — this determines whether step 1 is
   "delete the `CORSMiddleware` block" or "narrow its three parameters." Don't build
   the narrowed version speculatively if A turns out to be true; that's wasted surface
   area to maintain.
2. **Rate limiter module**: add `app/api/ratelimit.py` (same placement pattern as
   `app/api/auth.py`) holding:
   - An in-memory store: a `dict[str, list[float]]` or `dict[str, tuple[int, float]]`
     (key -> count + window-start timestamp) is enough for a fixed-window counter. No
     new dependency needed (§2.2) — plain `dict` + `time.monotonic()`.
   - A function like `check(key: str, *, limit: int, window_seconds: int = 60) -> bool`
     that increments the counter for `key`, resets it if the window has elapsed, and
     returns whether the caller is still under `limit`.
   - The two limit tiers from §2.2 as named constants (`DEFAULT_LIMIT_PER_MIN`,
     `MARKETING_LIMIT_PER_MIN`) so they're easy to find and tune later.
3. **Middleware**: register a `@app.middleware("http")` function, same shape as
   `_require_api_key` (`app/api/main.py`) and registered in the same relative position
   (inside the API-key guard, i.e. after it succeeds — no point counting requests that
   get rejected for a bad key anyway; or outside it, if you want failed-auth attempts to
   also count toward the limit as a brute-force guard — pick one and say why in the
   commit). Key the counter off `request.headers.get("X-API-Key", request.client.host)`
   so it degrades to per-IP in dev mode where no key is required.
   Everything else (except `/health`) gets the default tier.
4. **Concurrency note:** `dict` mutation from an async middleware running in a single
   `uvicorn` worker is safe without extra locking (no `await` between read-increment-write
   if written carefully) — don't add threading primitives that aren't needed for a
   single-process, single-event-loop server. If this service is ever run with multiple
   `uvicorn` workers (`--workers N`), each worker gets its own counter and the effective
   limit multiplies by N — call this out explicitly in the deploy docs (§6) so it's a
   known tradeoff, not a surprise.

---

## 4. Testing

- Extend `tests/test_api.py` (or a new `tests/test_ratelimit.py` if it's cleaner) with:
  - N+1 requests to a default-tier route within the window → the (N+1)th gets 429.
  - A request to `/health` past any limit → still 200 (stays exempt).
  - A 429 response includes `Retry-After`.
  - After the window elapses (mock/advance time rather than sleeping in a test — inject
    a clock or monkeypatch `time.monotonic`), the counter resets and requests succeed
    again.
- If CORS was narrowed (option B, §2.1): a preflight test confirming the now-explicit
  `allow_methods`/`allow_headers` still permit the calls the frontend actually makes,
  mirroring the existing `test_preflight_to_protected_route_still_gets_cors_headers` in
  `tests/test_api.py`.
- Confirm the existing test suite's default run (no `X-API-Key`, no explicit rate-limit
  config) stays unaffected — the default limits should be well above anything the test
  suite itself sends in a 60-second window; if any test loops enough to trip it, either
  raise the dev-mode default further or have that specific test use a distinct key.

---

## 5. Docs to update once this is built

- `README.md`'s env var table and `HANDOFF.md` §2 — document `CORS_ORIGINS`'s new
  fail-closed behavior (if option B) or its removal (if option A), and the two rate-limit
  constants/env-overrides if they're made configurable.
- `docs/02-backend-api-key-integration.md` — add a short note that the backend should
  expect occasional `429`s under abnormal conditions and treat them like the `401`
  guidance already there (§2.3 of that doc): log/alert, don't blind-retry in a tight loop
  (a tight retry loop is exactly the pattern this rate limit exists to catch, so a
  naive backend retry-on-429 could make its own problem worse).
- Whatever deploy manifest exists — note the `--workers N` caveat from §3 step 4 if this
  service is ever deployed with more than one worker process.

---

## 6. Explicitly out of scope / open questions for later

- **Real per-user rate limiting** — not possible at this layer without per-tenant
  identity, which `docs/01` §7 already flags as a bigger, separate feature. If that ever
  gets built (a key or token per `restaurantId`), per-tenant limits become possible and
  meaningful here too.
- **Shared/distributed rate-limit state** (Redis or similar) — only needed if this
  service is ever horizontally scaled across multiple hosts/instances; explicitly not
  built now (§2.2).
- **Dynamic/adaptive limits** (e.g. based on current GCP spend or LLM budget remaining) —
  a fixed, generously-set ceiling is the whole point of "safety net, not real rate
  limiting" (§1.2); a smarter, spend-aware system is a different, larger project.
- **CORS option A vs B decision** — must be made before implementation starts (§2.1); not
  pre-decided by this doc.
