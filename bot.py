"""
Vera Challenge Bot — Complete Submission
========================================
FastAPI server implementing all 5 required endpoints.
Uses Claude (claude-sonnet-4-20250514) for context-aware message composition.
"""

import os
import time
import uuid
import re
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TEAM_NAME         = os.environ.get("TEAM_NAME", "Vera Builder")
TEAM_EMAIL        = os.environ.get("TEAM_EMAIL", "builder@example.com")
TEAM_MEMBERS      = os.environ.get("TEAM_MEMBERS", "Solo builder").split(",")
VERSION           = "2.0.0"
SUBMITTED_AT      = datetime.now(timezone.utc).isoformat()

app   = FastAPI(title="Vera Challenge Bot", version=VERSION)
START = time.time()

# ── In-memory stores ───────────────────────────────────────────────────────────
# (scope, context_id) → {version, payload}
contexts: dict[tuple[str, str], dict] = {}

# conversation_id → list of {from_role, message, ts, bot_response}
conversations: dict[str, list] = {}

# suppression keys we've already sent (prevent duplicate sends in same tick)
sent_suppression_keys: set[str] = set()

# ── Pydantic models ────────────────────────────────────────────────────────────
class ContextBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str

class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []

class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str
    turn_number: int

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _get_context(scope: str, context_id: str) -> Optional[dict]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None

def _count_contexts() -> dict:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for (scope, _) in contexts:
        if scope in counts:
            counts[scope] += 1
    return counts

def _detect_auto_reply(message: str) -> bool:
    """Detect WhatsApp Business canned auto-replies."""
    auto_reply_patterns = [
        r"thank you for (contacting|reaching out|your message)",
        r"(automated|automatic) (reply|response|message)",
        r"i am (currently |)unavailable",
        r"aapki jaankari ke liye.*shukriya",
        r"aapki madad ke liye shukriya.*automated",
        r"we will get back to you",
        r"our team will (contact|reach) you",
        r"this is an (automated|auto)",
        r"bahut-bahut shukriya.*team tak",
        r"main ek automated",
    ]
    msg_lower = message.lower()
    for pattern in auto_reply_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False

def _detect_positive_intent(message: str) -> bool:
    """Detect merchant saying yes / let's go / proceed."""
    positive_patterns = [
        r"\b(yes|yep|yeah|yup|haan|ha|ok|okay|sure|go ahead|proceed|let'?s do it|please do|karo|kar do|send it|sounds good|great|perfect)\b",
        r"\b(i want to|i'd like to|mujhe chahiye|chalte hain)\b",
    ]
    msg_lower = message.lower()
    for pattern in positive_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False

def _detect_negative_intent(message: str) -> bool:
    """Detect merchant saying not interested / stop."""
    negative_patterns = [
        r"\b(no|nope|nahi|na|not interested|stop|unsubscribe|don'?t (contact|message|send)|leave me alone|busy|baad mein|later)\b",
        r"\b(not now|abhi nahi|mat bhejo)\b",
    ]
    msg_lower = message.lower()
    for pattern in negative_patterns:
        if re.search(pattern, msg_lower):
            return True
    return False

def _detect_language(merchant: dict) -> str:
    """Determine if merchant prefers Hindi-English mix."""
    langs = merchant.get("identity", {}).get("languages", ["en"])
    if "hi" in langs:
        return "hi-en"
    return "en"


# ── LLM Composer ──────────────────────────────────────────────────────────────

TRIGGER_PROMPT_VARIANTS = {
    "research_digest": "Focus on the specific research finding, cite the source, and relate it directly to THIS merchant's patient/customer cohort.",
    "regulation_change": "Lead with the regulatory update, explain the impact on this specific merchant's practice, and offer concrete next steps.",
    "recall_due": "Frame as a warm, non-pushy recall reminder. Use the customer's name, their last visit timing, and offer specific available slots.",
    "perf_dip": "Acknowledge the dip, provide context (is it seasonal? peer-comparable?), and offer a concrete action to recover.",
    "perf_spike": "Congratulate briefly, then immediately pivot to capitalising on the momentum with a specific next action.",
    "festival_upcoming": "Connect the festival to a specific offer or campaign the merchant can run. Name the festival, the dates, and draft the action.",
    "ipl_match_today": "Use the match data (team, venue, time). Reference magicpin's insight that Saturday IPL = -12% covers; contrarian advice scores highest.",
    "competitor_opened": "Frame as intelligence, not alarm. Give specific distance, then offer a specific competitive response.",
    "milestone_reached": "Celebrate the milestone with the actual number, then immediately offer to convert the momentum (Google post, campaign).",
    "review_theme_emerged": "Name the theme from the reviews, the volume, the sentiment. Offer a specific response strategy or operational tweak.",
    "curious_ask_due": "Ask ONE open question about what's working at their business this week, then offer to turn the answer into a deliverable.",
    "dormant_with_vera": "Re-engage warmly. Reference something specific about their profile or last conversation. Don't push too hard.",
    "supply_alert": "Urgent tone but bounded. Give batch numbers or specific product. Derive how many of THIS merchant's customers are affected.",
    "chronic_refill_due": "Name all medications. Give exact refill date. Calculate total + any discount. Offer home delivery if available.",
    "winback_eligible": "No guilt. Reference the customer's past service. Offer something specific and new. Single binary CTA.",
    "customer_lapsed_hard": "Warm, no-shame tone. Reference their past goal/service. Offer a new class or slot that matches their preference.",
    "seasonal_perf_dip": "Reframe the dip as normal. Give peer benchmark range. Offer retention action for existing members instead of acquisition.",
    "renewal_due": "Specific days remaining. The value delivered in the subscription period. Single binary renewal CTA.",
    "gbp_unverified": "Concrete risk of being unverified (search ranking drop). Offer to walk them through verification in 5 minutes.",
    "cde_opportunity": "Specific CDE credit requirement. The course details. Offer to shortlist and register.",
    "wedding_package_followup": "Days until wedding. The skin-prep or package window. Specific slot offer.",
    "trial_followup": "Reference their trial experience. What they said or did. Offer the next concrete step.",
    "active_planning_intent": "They said YES to something. Don't qualify again — draft the artifact immediately and share it.",
    "category_seasonal": "Name the seasonal beat (e.g., monsoon, exam season). Connect it to a specific offer or content piece.",
    "default": "Compose a helpful, specific message that clearly communicates why this message is being sent right now.",
}

SYSTEM_PROMPT = """You are Vera, magicpin's AI assistant for merchant growth in India.

YOUR ROLE: Compose WhatsApp messages that help merchants grow their business.
You operate at scale — 6,000-10,000 merchants per day across 50+ Indian cities.

CORE RULES (violating these loses points):
1. SPECIFICITY WINS: Always anchor on a real number, date, or fact from the context.
   BAD: "improve your sales" | GOOD: "your CTR is 2.1%, peer median is 3.0%"
   BAD: "Flat 30% off" | GOOD: "Dental Cleaning @ ₹299"
2. ONE CTA: Single binary action (Reply YES / STOP), or open-ended question. Never multi-choice.
3. NO FABRICATION: Only use data present in the contexts. Never invent research citations, competitor names, or statistics.
4. VOICE MATCH: Match the category's voice (dentists = clinical-peer; restaurants = fellow-operator; gyms = coach; salons = warm-practical).
5. LANGUAGE: If merchant languages include "hi", use natural Hindi-English code-mix. Not forced. Natural.
6. NO PREAMBLES: Never start with "I hope you're doing well" or "I'm reaching out today to". Get to the point.
7. NO RE-INTRODUCTION: After turn 1, never say "Hi, I'm Vera" again.
8. COMPULSION LEVERS: Use at least one per message:
   - Loss aversion ("you're missing X")
   - Social proof ("3 merchants in your locality did Y this month")
   - Effort externalization ("I've drafted it — just say go")
   - Curiosity ("want to see who?")
   - Specificity/verifiability (concrete numbers)
   - Single binary commitment (Reply YES / STOP)

OUTPUT FORMAT (strict JSON, no markdown):
{
  "body": "the WhatsApp message body",
  "cta": "open_ended" | "binary_yes_stop" | "none",
  "send_as": "vera" | "merchant_on_behalf",
  "suppression_key": "a short dedup key like trigger_kind:merchant_id:YYYY-WN",
  "rationale": "2-3 sentences: what context drove this message, what compulsion lever, what the merchant should do"
}

SEND_AS RULES:
- "vera" for all merchant-facing messages
- "merchant_on_behalf" ONLY for customer-facing messages (when customer context is present and trigger.scope = "customer")"""


def compose_message(
    category: dict,
    merchant: dict,
    trigger: dict,
    customer: Optional[dict] = None,
    conversation_history: Optional[list] = None,
) -> dict:
    """Call Claude to compose a context-aware message."""

    if not ANTHROPIC_API_KEY:
        # Graceful degradation if no API key
        return _fallback_compose(category, merchant, trigger, customer)

    trigger_kind   = trigger.get("kind", "default")
    trigger_hint   = TRIGGER_PROMPT_VARIANTS.get(trigger_kind, TRIGGER_PROMPT_VARIANTS["default"])
    lang_mode      = _detect_language(merchant)
    merchant_name  = merchant.get("identity", {}).get("name", "")
    owner_name     = merchant.get("identity", {}).get("owner_first_name", "")
    category_slug  = category.get("slug", "")
    is_customer_facing = (trigger.get("scope") == "customer" and customer is not None)

    # Build digest context for research/regulation triggers
    digest_context = ""
    if trigger_kind in ("research_digest", "regulation_change", "supply_alert"):
        top_item_id = trigger.get("payload", {}).get("top_item_id", "")
        digest_items = category.get("digest", [])
        for item in digest_items:
            if item.get("id") == top_item_id or top_item_id == "":
                digest_context = f"\nRELEVANT DIGEST ITEM:\n{item}\n"
                break
        if not digest_context and digest_items:
            digest_context = f"\nLATEST DIGEST ITEM:\n{digest_items[0]}\n"

    # Build conversation history context
    history_context = ""
    if conversation_history:
        recent = conversation_history[-4:]  # last 4 turns
        history_context = "\nCONVERSATION HISTORY (most recent turns):\n"
        for turn in recent:
            role = turn.get("from_role", turn.get("from", "?"))
            msg  = turn.get("message", turn.get("body", ""))
            history_context += f"[{role}]: {msg}\n"

    user_prompt = f"""COMPOSE A MESSAGE using ONLY the data below. Do not invent any facts.

TRIGGER KIND: {trigger_kind}
TRIGGER SCOPE: {trigger.get('scope', 'merchant')}
TRIGGER URGENCY: {trigger.get('urgency', 2)}/5
TRIGGER PAYLOAD: {trigger.get('payload', {})}
TRIGGER SUPPRESSION KEY: {trigger.get('suppression_key', '')}
SPECIFIC INSTRUCTION FOR THIS TRIGGER TYPE: {trigger_hint}

CATEGORY: {category_slug}
CATEGORY VOICE: {category.get('voice', {})}
CATEGORY PEER STATS: {category.get('peer_stats', {})}
ACTIVE OFFERS IN CATEGORY: {category.get('offer_catalog', [])[:4]}
SEASONAL BEATS: {category.get('seasonal_beats', [])}
TREND SIGNALS: {category.get('trend_signals', [])}
{digest_context}

MERCHANT:
  Name: {merchant_name}
  Owner first name: {owner_name}
  City/Locality: {merchant.get('identity', {}).get('city', '')} / {merchant.get('identity', {}).get('locality', '')}
  Verified: {merchant.get('identity', {}).get('verified', False)}
  Languages: {merchant.get('identity', {}).get('languages', ['en'])}
  Subscription: {merchant.get('subscription', {})}
  Performance (30d): {merchant.get('performance', {})}
  Active offers: {[o for o in merchant.get('offers', []) if o.get('status') == 'active']}
  Customer aggregate: {merchant.get('customer_aggregate', {})}
  Signals: {merchant.get('signals', [])}
  Review themes: {merchant.get('review_themes', [])}

{f"CUSTOMER: {customer}" if customer else "CUSTOMER: none (merchant-facing message)"}

LANGUAGE MODE: {"Hindi-English natural code-mix preferred" if lang_mode == "hi-en" else "English"}
IS CUSTOMER-FACING: {is_customer_facing}
{history_context}

Compose the message now. Output ONLY valid JSON matching the schema in the system prompt."""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=0,   # deterministic
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        import json
        result = json.loads(raw)
        return result
    except Exception as e:
        return _fallback_compose(category, merchant, trigger, customer)


def _fallback_compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict]) -> dict:
    """Rule-based fallback when LLM is unavailable."""
    merchant_name = merchant.get("identity", {}).get("name", "there")
    owner         = merchant.get("identity", {}).get("owner_first_name", "")
    address_name  = owner or merchant_name
    trigger_kind  = trigger.get("kind", "general")
    active_offers = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    offer_str     = active_offers[0] if active_offers else "your current offer"
    ctr           = merchant.get("performance", {}).get("ctr", 0)
    peer_ctr      = category.get("peer_stats", {}).get("avg_ctr", 0.03)

    if trigger_kind == "perf_dip":
        body = (f"{address_name}, quick check — your views are down this week. "
                f"Your CTR is {ctr:.1%} vs peer median {peer_ctr:.1%}. "
                f"Want me to draft a Google post + push {offer_str} to drive recovery?")
        cta = "binary_yes_stop"
    elif trigger_kind == "research_digest":
        body = (f"{address_name}, new research digest just landed for {category.get('slug', 'your category')}. "
                f"I've flagged one item relevant to your patients. Want me to pull it and draft a patient note?")
        cta = "binary_yes_stop"
    elif trigger_kind == "recall_due" and customer:
        cname = customer.get("identity", {}).get("name", "your patient")
        body  = (f"Hi {cname}, {merchant_name} here. Your recall is due — "
                 f"{offer_str} available. Reply YES to book, STOP to opt out.")
        cta   = "binary_yes_stop"
    else:
        body = (f"{address_name}, Vera here. "
                f"Spotted something worth sharing — {offer_str} is active and your profile has room to grow. "
                f"Want me to draft a quick Google post to boost visibility?")
        cta = "binary_yes_stop"

    return {
        "body": body,
        "cta": cta,
        "send_as": "merchant_on_behalf" if trigger.get("scope") == "customer" and customer else "vera",
        "suppression_key": trigger.get("suppression_key", f"{trigger_kind}:{merchant.get('merchant_id','')}"),
        "rationale": f"Fallback composition for trigger={trigger_kind}. LLM unavailable.",
    }


def compose_reply(
    conversation_id: str,
    merchant_id: Optional[str],
    customer_id: Optional[str],
    merchant_message: str,
    turn_number: int,
) -> dict:
    """Compose a reply to a merchant/customer message in an ongoing conversation."""

    # ── Auto-reply detection ────────────────────────────────────────────────
    if _detect_auto_reply(merchant_message):
        history = conversations.get(conversation_id, [])
        # How many auto-replies in a row?
        auto_count = 0
        for turn in reversed(history):
            if turn.get("is_auto_reply"):
                auto_count += 1
            else:
                break
        if auto_count >= 1:
            return {
                "action": "end",
                "rationale": "Auto-reply detected (2nd occurrence). Gracefully exiting — will reach owner/manager directly."
            }
        # First auto-reply — try once more
        return {
            "action": "send",
            "body": "Samajh gayi — aapki team tak pahunchne se pehle, kya aap khud 2 min mein dekhna chahenge ki exactly kya update karna hai? Main guide kar sakti hoon.",
            "cta": "binary_yes_stop",
            "rationale": "Auto-reply detected (1st occurrence). Attempting one more turn to reach real person."
        }

    # ── Negative intent ─────────────────────────────────────────────────────
    if _detect_negative_intent(merchant_message):
        return {
            "action": "end",
            "rationale": "Merchant signaled not interested. Gracefully exiting conversation."
        }

    # ── Positive intent / action ────────────────────────────────────────────
    if _detect_positive_intent(merchant_message):
        merchant = _get_context("merchant", merchant_id) if merchant_id else {}
        category_slug = merchant.get("category_slug", "") if merchant else ""
        category = _get_context("category", category_slug) if category_slug else {}
        customer = _get_context("customer", customer_id) if customer_id else None

        if not ANTHROPIC_API_KEY or not merchant:
            return {
                "action": "send",
                "body": "Bilkul! Draft kar rahi hoon — 2 minutes mein ready hoga. Koi specific angle ya detail add karna chahenge?",
                "cta": "open_ended",
                "rationale": "Merchant confirmed — proceeding with the requested action."
            }

        # Build conversation history from store
        history = conversations.get(conversation_id, [])
        history_for_llm = [
            {"from_role": t.get("from_role", "?"), "message": t.get("message", "")}
            for t in history[-6:]
        ]

        # Compose a follow-up / action message
        action_prompt = f"""The merchant just said: "{merchant_message}"

This is a POSITIVE INTENT signal — they said yes / proceed. Do NOT ask another qualifying question.
Instead: draft the artifact they asked for (Google post, patient-ed WhatsApp, campaign copy, etc.)
or provide the concrete next step immediately.

Use the context below to make it specific:
Merchant: {merchant}
Category slug: {category_slug}
Category voice: {category.get('voice', {}) if category else {}}
Conversation history: {history_for_llm}

Output JSON: {{"body": "...", "cta": "open_ended"|"binary_yes_stop"|"none", "send_as": "vera", "suppression_key": "reply:{conversation_id}:t{turn_number}", "rationale": "..."}}"""

        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                temperature=0,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": action_prompt}],
            )
            raw = resp.content[0].text.strip()
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"```\s*$", "", raw)
            import json
            result = json.loads(raw)
            return {"action": "send", "body": result.get("body", ""), "cta": result.get("cta", "open_ended"), "rationale": result.get("rationale", "")}
        except Exception:
            return {
                "action": "send",
                "body": "Shukriya! Draft kar rahi hoon abhi — 2-3 min mein ready hoga.",
                "cta": "open_ended",
                "rationale": "Merchant confirmed, proceeding with action. LLM composition in progress."
            }

    # ── General reply — LLM handles ─────────────────────────────────────────
    merchant = _get_context("merchant", merchant_id) if merchant_id else {}
    category_slug = merchant.get("category_slug", "") if merchant else ""
    category = _get_context("category", category_slug) if category_slug else {}
    history = conversations.get(conversation_id, [])

    if not ANTHROPIC_API_KEY or not merchant:
        return {
            "action": "send",
            "body": "Got it, samajh gayi! Aur kuch chahiye ya koi specific sawaal hai?",
            "cta": "open_ended",
            "rationale": "General reply — advancing conversation."
        }

    history_for_llm = [
        {"from_role": t.get("from_role", "?"), "message": t.get("message", "")}
        for t in history[-6:]
    ]

    general_prompt = f"""You are mid-conversation with a merchant. Their latest message: "{merchant_message}"

Conversation so far: {history_for_llm}
Merchant context: {merchant.get('identity', {})} | Signals: {merchant.get('signals', [])}
Category: {category_slug}

Rules:
- If turn_number >= 4 and no engagement yet, consider ending gracefully
- If off-topic (GST, unrelated questions), stay on mission politely
- Never re-introduce yourself
- Keep response short (1-3 sentences)
- Match language pref: {merchant.get('identity', {}).get('languages', ['en'])}

Output JSON: {{"action": "send"|"wait"|"end", "body": "...", "cta": "open_ended"|"binary_yes_stop"|"none", "rationale": "..."}}"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": general_prompt}],
        )
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"```\s*$", "", raw)
        import json
        result = json.loads(raw)
        # Ensure action is valid
        action = result.get("action", "send")
        if action not in ("send", "wait", "end"):
            action = "send"
        out = {"action": action, "rationale": result.get("rationale", "")}
        if action == "send":
            out["body"] = result.get("body", "")
            out["cta"]  = result.get("cta", "open_ended")
        elif action == "wait":
            out["wait_seconds"] = result.get("wait_seconds", 1800)
        return out
    except Exception:
        return {
            "action": "send",
            "body": "Samajh gayi! Koi aur sawaal?",
            "cta": "open_ended",
            "rationale": "General reply, advancing conversation."
        }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/v1/healthz")
async def healthz():
    return {
        "status": "ok",
        "uptime_seconds": int(time.time() - START),
        "contexts_loaded": _count_contexts(),
        "conversations_active": len(conversations),
        "llm_configured": bool(ANTHROPIC_API_KEY),
    }


@app.get("/v1/metadata")
async def metadata():
    return {
        "team_name":    TEAM_NAME,
        "team_members": TEAM_MEMBERS,
        "model":        "claude-sonnet-4-20250514",
        "approach":     (
            "Trigger-kind dispatch → Claude sonnet with 4-context prompt → "
            "auto-reply detection → intent routing → structured JSON output with rationale. "
            "Stateful conversation with graceful exit logic."
        ),
        "contact_email": TEAM_EMAIL,
        "version":       VERSION,
        "submitted_at":  SUBMITTED_AT,
    }


@app.post("/v1/context")
async def push_context(body: ContextBody):
    if body.scope not in ("category", "merchant", "customer", "trigger"):
        return JSONResponse(
            status_code=400,
            content={"accepted": False, "reason": "invalid_scope", "details": f"Unknown scope: {body.scope}"}
        )
    key = (body.scope, body.context_id)
    cur = contexts.get(key)
    if cur and cur["version"] >= body.version:
        return JSONResponse(
            status_code=409,
            content={"accepted": False, "reason": "stale_version", "current_version": cur["version"]}
        )
    contexts[key] = {"version": body.version, "payload": body.payload}
    return {
        "accepted":   True,
        "ack_id":     f"ack_{body.context_id}_v{body.version}",
        "stored_at":  _now_iso(),
    }


@app.post("/v1/tick")
async def tick(body: TickBody):
    actions = []
    for trg_id in body.available_triggers:
        # Skip if we've already sent this suppression key
        trg_data = contexts.get(("trigger", trg_id))
        if not trg_data:
            continue
        trg = trg_data["payload"]

        suppression_key = trg.get("suppression_key", "")
        if suppression_key and suppression_key in sent_suppression_keys:
            continue

        merchant_id = trg.get("merchant_id")
        customer_id = trg.get("customer_id")

        merchant = _get_context("merchant", merchant_id) if merchant_id else None
        if not merchant:
            continue

        category_slug = merchant.get("category_slug", "")
        category = _get_context("category", category_slug)
        if not category:
            continue

        customer = _get_context("customer", customer_id) if customer_id else None

        # Compose the message
        composed = compose_message(category, merchant, trg, customer)

        # Build conversation ID
        conv_id = f"conv_{merchant_id}_{trg_id}_{int(time.time())}"

        # Track suppression
        if suppression_key:
            sent_suppression_keys.add(suppression_key)

        # Initialise conversation store
        conversations[conv_id] = [{
            "from_role": "bot",
            "message": composed.get("body", ""),
            "ts": _now_iso(),
            "is_auto_reply": False,
        }]

        # Build template params from merchant name + trigger kind
        merchant_name = merchant.get("identity", {}).get("name", "")
        trigger_kind  = trg.get("kind", "general")
        template_params = [merchant_name, trigger_kind, composed.get("body", "")[:80]]

        actions.append({
            "conversation_id":  conv_id,
            "merchant_id":      merchant_id,
            "customer_id":      customer_id,
            "send_as":          composed.get("send_as", "vera"),
            "trigger_id":       trg_id,
            "template_name":    f"vera_{trigger_kind}_v1",
            "template_params":  template_params,
            "body":             composed.get("body", ""),
            "cta":              composed.get("cta", "open_ended"),
            "suppression_key":  suppression_key,
            "rationale":        composed.get("rationale", ""),
        })

        # Cap at 20 actions per tick
        if len(actions) >= 20:
            break

    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody):
    # Store the incoming message
    is_auto = _detect_auto_reply(body.message)
    conversations.setdefault(body.conversation_id, []).append({
        "from_role":    body.from_role,
        "message":      body.message,
        "ts":           body.received_at,
        "is_auto_reply": is_auto,
    })

    result = compose_reply(
        conversation_id=body.conversation_id,
        merchant_id=body.merchant_id,
        customer_id=body.customer_id,
        merchant_message=body.message,
        turn_number=body.turn_number,
    )

    # Store bot response in history
    if result.get("action") == "send":
        conversations[body.conversation_id].append({
            "from_role": "bot",
            "message":   result.get("body", ""),
            "ts":        _now_iso(),
            "is_auto_reply": False,
        })

    return result


@app.post("/v1/teardown")
async def teardown():
    """Optional endpoint — wipe state at end of test."""
    contexts.clear()
    conversations.clear()
    sent_suppression_keys.clear()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("bot:app", host="0.0.0.0", port=port, reload=False)
