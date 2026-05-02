# Vera Challenge Bot

A context-aware WhatsApp message engine for magicpin merchant growth — built with Claude (claude-sonnet-4-20250514) as the composition core, trigger-kind dispatch, and production-grade conversation handling.

---

## Live Endpoint

```
https://<your-render-url>.onrender.com
```

All 5 required endpoints are live:

| Endpoint | Purpose |
|---|---|
| `GET /v1/healthz` | Liveness probe |
| `GET /v1/metadata` | Bot identity |
| `POST /v1/context` | Receive context pushes (idempotent by version) |
| `POST /v1/tick` | Periodic wake-up; bot initiates proactive sends |
| `POST /v1/reply` | Handle merchant/customer replies |

---

## Architecture

```
Judge pushes context → /v1/context
                              ↓
                   In-memory context store
                   (category / merchant / customer / trigger)
                              ↓
Judge calls /v1/tick → Trigger dispatcher
                              ↓
                   Trigger-kind router
                   (24 trigger kinds → specific prompt variant)
                              ↓
                   Claude composer
                   (4-context prompt → JSON output)
                              ↓
                   Suppression dedup → actions[]
                              ↓
Judge sends merchant reply → /v1/reply
                              ↓
                   Intent classifier
                   (auto-reply? positive? negative? general?)
                              ↓
                   Reply composer (Claude, with history)
                              ↓
                   send / wait / end
```

---

## Approach

### 1. Trigger-kind dispatch

Every trigger kind maps to a specific prompt variant with a concrete instruction tailored to that kind. For example:

- `research_digest` → "Cite the source, relate to THIS merchant's patient cohort"
- `ipl_match_today` → "Reference magicpin insight: Saturday IPL = -12% covers; give contrarian recommendation"
- `supply_alert` → "Give batch numbers, derive how many of THIS merchant's customers are affected"
- `active_planning_intent` → "They said YES — don't qualify again, draft the artifact immediately"

This means a `research_digest` for Dr. Meera produces clinical, source-cited output; the same engine for a `winback_eligible` trigger for a gym produces a no-shame, goal-specific message. Same code, 24 different conversation shapes.

### 2. Four-context composition

Every Claude call receives all four contexts simultaneously — category voice/vocabulary, merchant-specific performance and signals, trigger payload, and customer context when present. Claude is instructed to only use data present in the contexts, never fabricate statistics or citations.

Key rules enforced in the system prompt:
- Specificity over generics ("CTR 2.1% vs peer 3.0%" beats "improve your profile")
- Service+price anchors ("Dental Cleaning @ ₹299") over percentage discounts
- Single binary CTA per message
- Domain vocabulary from category.voice

### 3. Language matching

Merchant `identity.languages` is checked at composition time. If Hindi is present, the prompt instructs Claude to use natural Hindi-English code-mix — not forced, not translated, but the way a real Vera message would read.

### 4. Conversation intelligence

**Auto-reply detection**: Regex patterns match common WhatsApp Business canned replies in both English and Hindi. First detection → one more attempt. Second detection → graceful exit.

**Intent routing**: 
- Positive intent (yes/ok/haan/proceed) → skip qualification, draft the artifact immediately
- Negative intent (no/nahi/stop/not interested) → graceful exit
- General replies → LLM handles with conversation history context

**Anti-repetition**: Suppression keys are tracked in memory per test session; the same key won't fire twice in the same run.

### 5. Graceful degradation

If the LLM call fails or times out, a rule-based fallback composer produces a context-aware message using merchant name, active offers, and trigger kind. The bot never returns an empty body.

---

## What additional context would have helped most

1. **Real merchant conversation history** beyond the 2-turn sample — knowing whether a merchant typically engages, which topics they respond to, and their preferred message length would sharpen the opening hook significantly.

2. **Live slot availability for recall/appointment triggers** — the customer-facing recall messages are strongest when they name real available slots. With a slot-lookup tool, the bot could generate booking-ready messages rather than asking the merchant to confirm slots separately.

3. **Category-specific regulatory calendar** — for pharmacies and dentists, knowing *which* compliance deadlines are upcoming (DCI renewal, drug license renewal) would make the regulatory triggers much more specific.

4. **Per-merchant A/B history** — knowing which message shapes have historically gotten this merchant to reply would let the bot adapt its hook selection per merchant, not just per category.

---

## Tradeoffs

| Decision | Why |
|---|---|
| In-memory state (no Redis) | Sufficient for the 60-minute test window; no external dependencies to fail |
| Temperature = 0 | Challenge requires determinism; same inputs → same output every run |
| 24 trigger-kind prompt variants | Avoids one-size-fits-all prompt that scores poorly on category fit |
| Fallback composer | Bot never silently fails; degraded output is better than a timeout |
| Auto-reply detection before LLM | Saves LLM latency on a detectable pattern; avoids burning 2-3 turns on canned replies |

---

## Local development

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env: add ANTHROPIC_API_KEY, your name, email

# 3. Run
uvicorn bot:app --reload --port 8080

# 4. Test with judge simulator
export BOT_URL=http://localhost:8080
python judge_simulator.py   # from the challenge zip
```

---

## Deployment (Render — free tier, 10 min)

1. Push this folder to a GitHub repo
2. Go to [render.com](https://render.com) → New Web Service → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn bot:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `ANTHROPIC_API_KEY`, `TEAM_NAME`, `TEAM_EMAIL`
6. Deploy → copy the public URL → submit to magicpin portal

The `render.yaml` in this repo automates steps 2-4.
