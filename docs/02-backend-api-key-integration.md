# 02 — Backend changes needed for the AI service's new API key auth

**Audience:** whoever (or whichever AI agent) maintains the RestoMind NestJS backend that
calls this AI service. This doc is written to be handed to that agent directly — it
describes only what changed on the AI service's side and exactly what the backend must
do in response. It does not require reading the AI service's own codebase.

**Status:** the AI service side is already implemented and merged (see
`docs/01-api-key-hardening.md` for the full design). This doc is the backend's half of
that rollout.

---

## 1. What changed, in one paragraph

The AI service (the "prediction-model" / "Bakery Demand & Surplus AI" service you call
over HTTP) used to have **no authentication on almost every route**, and a narrow,
easy-to-forget shared-secret check (`X-RestoMind-Key` header, gated by an
`AI_SHARED_SECRET` env var that was usually unset) on only the `/integration/restomind/*`
routes. That's been replaced: **every route now requires a single API key**, sent as a
header on every request. The old header name and env var are gone — there is no
backward-compatible fallback.

---

## 2. Exactly what the backend needs to do

### 2.1 Send a new header on every call to the AI service

Change every outgoing HTTP call this backend makes to the AI service (all of
`/forecast/*`, `/data/ingest`, `/model/status`, `/alerts/waste-prevention`,
`/surplus/detect`, `/marketing/*`, and all `/integration/restomind/*` routes) to include:

```
X-API-Key: <the raw key — see §2.2 for where it comes from>
```

`/health` is the only route that does **not** require this header — leave health checks
as they are.

If any existing code sends `X-RestoMind-Key`, remove/replace it. That header name is no
longer read by the AI service at all (not even as a fallback) — requests using only the
old header will now get `401 Unauthorized`.

### 2.2 Where the key itself comes from

This is a **shared secret coordination step between the two teams/services**, not
something the backend generates unilaterally:

1. Whoever owns the AI service's deployment generates the raw key (a random 256-bit
   value) and gives it to the backend team out-of-band — a secrets manager, a password
   vault share, anything except plaintext chat/email/commit.
2. The AI service's own deployment config stores only a **hash** of that key
   (`API_KEY_HASH` env var, SHA-256 of the raw key) — it never stores the raw value
   itself. This is mentioned here for context; it is **not** something the backend needs
   to replicate. The backend stores and sends the **raw** key, not a hash of it.
3. The backend stores the raw key exactly like it stores any other outbound-service
   credential (its Meta/Facebook token, DB credentials, etc.) — in whatever secrets
   manager or environment-injected config this backend already uses. Never commit it,
   never log it, never put it in a frontend-reachable env var.

If you don't already have this raw key, that's a blocker — ask whoever is deploying the
AI service for it before wiring this up. Do not invent a placeholder value and ship it;
requests with a wrong key get a real `401`, not a warning.

### 2.3 Handle 401s explicitly

Any call to the AI service can now fail with:

```json
{"error": "unauthorized", "detail": "Missing or invalid X-API-Key"}
```
HTTP status `401`.

This should be treated as a **configuration/credential problem**, not a normal
application-level error to surface to end users. Concretely:

- If this happens in production, it almost certainly means the key configured on the
  backend's side doesn't match what the AI service expects (typo, stale value after a
  rotation, wrong environment's secret used) — alert/log loudly, don't silently retry
  forever or swallow it into a generic "AI service unavailable" message that hides the
  real cause.
- Don't build automatic retry-with-backoff logic that treats a 401 like a transient
  5xx — a wrong key will never succeed on retry, and hammering the endpoint with a bad
  key doesn't help. Retries are fine for actual transient failures (timeouts, 5xx), just
  not for 401.

### 2.4 Key rotation is manual, coordinate before it happens

There is no automated rotation. When the AI service's key is rotated (see
`docs/01-api-key-hardening.md` §3.4), it will be a **coordinated, manual, two-sided
change**:

1. AI service side gets the new key's hash deployed first (it will now accept the new
   key).
2. The backend needs to update its stored raw key to match, then redeploy/restart so it
   starts sending the new value.
3. There will be a brief window between step 1 and step 2 where the *old* key the
   backend is still sending gets rejected (401) — this is expected during a rotation, not
   a bug. Coordinate the timing with whoever owns the AI service's deployment rather than
   discovering it via a spike in 401s.

There is no schedule for this — it only happens on-demand (e.g. suspected leak). No
backend-side automation is needed for it now; a follow-up will be flagged separately if
that changes.

### 2.5 Nothing else about the request/response shapes changed

This is purely an auth header addition. No endpoint URLs, request bodies, or response
shapes changed. If existing integration code already works today (modulo the missing
header), it will keep working once the header is added — no other client-side changes
are needed.

---

## 3. Checklist for whoever implements this

- [ ] Obtain the raw API key from the AI service's deployment owner (§2.2) — do not
      proceed without it.
- [ ] Store it in this backend's existing secrets mechanism (never a committed file, never
      a client-reachable env var).
- [ ] Add `X-API-Key: <key>` to every outgoing request to the AI service (all routes
      except `/health`).
- [ ] Remove any code still sending `X-RestoMind-Key` — it does nothing now.
- [ ] Verify with a real call against a non-production AI service instance: request
      without the header → 401; request with the header → 200 (matching pre-existing
      behavior).
- [ ] Add/confirm logging or alerting on 401 responses from the AI service specifically
      (§2.3), distinct from other error handling.
- [ ] Note internally who to contact / what process to follow when the AI service rotates
      this key (§2.4) — this backend needs to be ready to update its stored key on short
      notice when that happens.

---

## 4. Reference: what the AI service's `.env.example` looks like now

For context only — this is the AI service's own config, not something the backend sets:

```
REQUIRE_API_KEY=false
API_KEY_HASH=
```

In any deployed (non-local) environment, the AI service's owner sets `REQUIRE_API_KEY=true`
and `API_KEY_HASH=<sha256 of the raw key>`. Until that's turned on there, the AI service
runs in dev mode and does not actually enforce the key — but backend integration code
should send the header regardless, so nothing breaks the moment enforcement is switched
on.
