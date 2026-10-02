"""AI-drafted retention messages: the last mile from "who is at risk and why" to "what we say to them".

Design for trust, not just fluency:
  * Grounded: the prompt contains ONLY the member's SHAP drivers, persona, preferred category, tier and the
    recommended action. The model is told not to invent facts.
  * Privacy-safe: no name, e-mail, customer ID, age or city leaves the system; only behavioural signals.
  * Guardrailed: every draft is checked (length, offer cap, no mention of churn / scores / models / tracking).
    A failed draft is retried once, then replaced by a deterministic template.
  * Works offline: with no GROQ_API_KEY the template engine writes every message, so the project always runs.

    python -m src.genai --n 60            # draft for the top-60 priority members (Groq if a key is set)
    python -m src.genai --template-only   # force templates
"""
import argparse
import hashlib
import json
import os
import re
import time

import pandas as pd

from . import config as C

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b"
OUT = C.OUTPUTS / "retention_messages.csv"

CHANNEL = {"Personalised coupon": "App push + email", "Proactive service call": "Agent call script",
           "Win-back offer": "Email"}
OFFER = {"Personalised coupon": C.COUPON_COST, "Win-back offer": C.WINBACK_COST, "Proactive service call": C.COUPON_COST}
BANNED = re.compile(r"\b(churn\w*|probabilit\w*|risk score|model|algorithm|machine learning|AI|predict\w*|tracking|"
                    r"noticed|we see that|data shows|gift ?card|voucher code|promo code|discount code|use code)\b", re.I)
INVENTED_CODE = re.compile(r"\b[A-Z]{3,}\d{1,4}\b")          # e.g. WELCOME5: codes we never issued
MAX_WORDS = {"App push + email": 70, "Email": 80, "Agent call script": 110}

SYSTEM = """You write customer-retention messages for FreshBasket, a neighbourhood grocery chain with a loyalty program.
Rules (all mandatory):
- Use ONLY the facts given. Do not invent purchases, dates, names, products or prices.
- Never mention churn, risk, scores, models, AI, prediction or data tracking. Sound like a helpful store team, not an algorithm.
- Never use the customer's name (you do not have it). Address them as "you".
- Any offer must be at most the stated offer budget in dollars, and must relate to the preferred category when one is given.
- The offer is ALREADY loaded onto the member's loyalty account: say so. Never invent promo codes, links, gift cards, vouchers,
  expiry dates or store names.
- Never describe the member's behaviour back to them ("we noticed you visit less", "you haven't opened our emails"). Be welcoming instead.
- "support_history" gives counts of recent support tickets and whether a formal complaint was filed. If there are recent tickets or a
  complaint, the message (especially a call script) must open by acknowledging that they recently contacted us with an issue,
  thank them for raising it, and ask whether it has been fully resolved. You do NOT know what the issue was: never guess or
  describe it. If there are no tickets and no complaint, do not apologise or imply anything went wrong.
- Warm, specific, one clear call to action. No emojis, no pressure, no exclamation-mark spam (max one).
- An "Agent call script" is an OUTBOUND call: the store is calling the member. Do not say they reached out or called us
  unless support_history shows tickets.
- For an "Agent call script": write 3-4 short lines the agent says (one per line). Open with an acknowledgement of their support
  request if support issues are listed, otherwise with a friendly check-in; ask an open question; mention the loaded offer; close politely.
Return JSON only: {"subject": "<short subject or call opener>", "message": "<the message>"}"""


# --------------------------------------------------------------------------- key handling
def api_key() -> str | None:
    """GROQ_API_KEY from the environment, or from a git-ignored .env file in the project root."""
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"].strip()
    env = C.ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


# --------------------------------------------------------------------------- context & templates
def _support_history(row) -> dict:
    """Recent support contact, as facts the agent can acknowledge (counts only; ticket contents are never available)."""
    get = lambda c: row.get(c) if c in row and pd.notna(row.get(c)) else 0
    return {"support_tickets_last_3_months": int(get("tickets_l3")), "support_tickets_last_6_months": int(get("tickets_6m")),
            "formal_complaint_last_3_months": bool(get("complaint_l3"))}


def has_support_issue(ctx: dict) -> bool:
    s = ctx.get("support_history", {})
    return (s.get("support_tickets_last_6_months", 0) > 0 or s.get("formal_complaint_last_3_months", False)
            or any(w in " ".join(ctx["why_flagged"]).lower() for w in ("ticket", "complaint")))


def context(row: pd.Series) -> dict:
    """The ONLY information the LLM sees (no identifiers, no demographics)."""
    action = row.recommended_action
    return {"channel": CHANNEL.get(action, "Email"), "recommended_action": action,
            "support_history": _support_history(row),
            "persona": row.get("persona", "") if pd.notna(row.get("persona", "")) else "",
            "membership_tier": row.tier, "preferred_category": row.preferred_category,
            "why_flagged": str(row.top_3_drivers).split("; "),
            "offer_budget_usd": OFFER.get(action, C.COUPON_COST)}


def template(ctx: dict) -> dict:
    cat, tier, budget = ctx["preferred_category"], ctx["membership_tier"], ctx["offer_budget_usd"]
    support = has_support_issue(ctx)
    complaint = ctx.get("support_history", {}).get("formal_complaint_last_3_months", False)
    if ctx["recommended_action"] == "Proactive service call":
        lines = [("Hi, this is the FreshBasket store team. I'm following up on the concern you raised with us recently, and thank you for telling us about it."
                  if complaint else "Hi, this is the FreshBasket store team. I'm calling about your recent support requests, and I'm sorry we didn't get it right first time.")
                 if support else "Hi, this is the FreshBasket store team, checking in on how your recent shopping trips have gone.",
                 "Is there anything still unresolved that I can fix for you today?",
                 f"As a thank-you for your patience as a {tier} member, we've added ${budget:.0f} off your next {cat.lower()} shop.",
                 "Is there anything else that would make FreshBasket work better for you?"]
        return {"subject": "Service follow-up call", "message": " ".join(lines)}
    if ctx["recommended_action"] == "Win-back offer":
        return {"subject": "We've missed you at FreshBasket",
                "message": f"It's been a while since your last visit, and we'd love to see you back. Here's ${budget:.0f} off your next "
                           f"{cat.lower()} shop, valid for 30 days. Your {tier} benefits are waiting for you."}
    return {"subject": f"A little something for your next {cat.lower()} shop",
            "message": f"Thanks for being a {tier} member. We've added ${budget:.0f} off {cat.lower()} to your account for your next visit. "
                       "Open the app to see this week's picks, chosen around what you buy most."}


def check(draft: dict, ctx: dict) -> list[str]:
    """Guardrails. Returns a list of violations (empty = pass)."""
    text = f"{draft.get('subject', '')} {draft.get('message', '')}"
    problems = []
    if not draft.get("message"):
        problems.append("empty")
    if BANNED.search(text):
        problems.append(f"banned term: {BANNED.search(text).group(0)}")
    if len(draft.get("message", "").split()) > MAX_WORDS.get(ctx["channel"], 80):
        problems.append("too long")
    amounts = [float(a) for a in re.findall(r"\$(\d+(?:\.\d+)?)", text)]
    if any(a > ctx["offer_budget_usd"] + 1e-9 for a in amounts):
        problems.append("offer above budget")
    if re.search(r"\d+\s?%", text):
        problems.append("percentage offer (budget is in dollars)")
    if INVENTED_CODE.search(draft.get("message", "")):
        problems.append(f"invented code: {INVENTED_CODE.search(draft.get('message', '')).group(0)}")
    support = has_support_issue(ctx)
    ACK = r"\b(raised|reach(?:ed|ing) out|contact(?:ed)? us|concerns?|issues?|\w*resolved|requests?|feedback|got in touch)\b"
    if support and ctx["recommended_action"] == "Proactive service call" and not re.search(ACK, text, re.I):
        problems.append("call script ignores the member's recent support contact")
    if not support and re.search(r"\b(sorry|apolog\w*|tough time|inconvenience|reached out|you called|your request)\b", text, re.I):
        problems.append("implies a support issue that is not in the data")
    return problems


# --------------------------------------------------------------------------- LLM
def _call_groq(ctx: dict, key: str, model: str, feedback: str = "") -> dict:
    import httpx
    user = "Write the message for this member:\n" + json.dumps(ctx, indent=1)
    if feedback:
        user += f"\n\nYour previous draft broke these rules: {feedback}. Fix them."
    body = {"model": model, "temperature": 0.4, "max_tokens": 600, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    for attempt in range(4):
        r = httpx.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
        if r.status_code == 429:                       # free-tier rate limit: back off and retry
            time.sleep(float(r.headers.get("retry-after", 5 * (attempt + 1))))
            continue
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return json.loads(content[content.find("{"): content.rfind("}") + 1])
    raise RuntimeError("Groq rate limit persisted")


def draft(row: pd.Series, key: str | None = None, model: str | None = None) -> dict:
    """One member -> {channel, subject, message, source, guardrail_pass, guardrail_notes}."""
    ctx = context(row)
    key = key if key is not None else api_key()
    model = model or os.environ.get("GROQ_MODEL", DEFAULT_MODEL)
    out, notes, source = None, [], "template"
    if key:
        try:
            out = _call_groq(ctx, key, model)
            notes = check(out, ctx)
            if notes:
                out = _call_groq(ctx, key, model, feedback="; ".join(notes))
                notes = check(out, ctx)
            source = f"groq:{model}"
        except Exception as e:                          # network / auth / parsing: never break the pipeline
            notes, out = [f"LLM error: {type(e).__name__}"], None
    if out is None or notes:
        fallback_reason = "; ".join(notes)
        out, source = template(ctx), "template" + (f" (fallback: {fallback_reason})" if fallback_reason else "")
        notes = check(out, ctx)
    return {"channel": ctx["channel"], "subject": out.get("subject", ""), "message": out.get("message", ""),
            "source": source, "guardrail_pass": not notes, "guardrail_notes": "; ".join(notes)}


def _fingerprint(row) -> str:
    sup = _support_history(row)
    return hashlib.sha1(f"{row.recommended_action}|{row.top_3_drivers}|{row.get('persona', '')}|{sorted(sup.items())}".encode()).hexdigest()[:12]


def run(actions: pd.DataFrame | None = None, n: int = 60, template_only: bool = False, refresh: bool = False,
        pause: float = 2.2) -> pd.DataFrame:
    """Draft messages for the top-n priority members with a paid action; cached by input fingerprint."""
    actions = actions if actions is not None else pd.read_csv(C.OUTPUTS / "retention_action_list.csv")
    paid = actions[actions.recommended_action != "Monitor (no paid action)"].sort_values("priority_rank")
    per_action = -(-n // paid.recommended_action.nunique())          # balanced across coupon / call / win-back
    todo = paid.groupby("recommended_action", group_keys=False).head(per_action).sort_values("priority_rank").head(n)
    key = None if template_only else api_key()
    cache = pd.read_csv(OUT) if OUT.exists() and not refresh else pd.DataFrame()
    rows = []
    for _, r in todo.iterrows():
        fp = _fingerprint(r)
        hit = cache[(cache.customer_id == r.customer_id) & (cache.fingerprint == fp)] if len(cache) else cache
        if len(hit):
            cached = hit.iloc[0].to_dict()
            still_ok = not check({"subject": cached.get("subject", ""), "message": cached.get("message", "")}, context(r))
            if still_ok and (template_only or not str(cached["source"]).startswith("template") or not key):
                rows.append(cached)                      # cached drafts must still pass the CURRENT guardrails
                continue
        d = draft(r, key=key or "")
        rows.append({"priority_rank": r.priority_rank, "customer_id": r.customer_id, "recommended_action": r.recommended_action,
                     "persona": r.get("persona", ""), "top_3_drivers": r.top_3_drivers, "fingerprint": fp, **d})
        if key and d["source"].startswith("groq"):
            time.sleep(pause)                            # stay under free-tier requests-per-minute
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    llm = out.source.str.startswith("groq").sum()
    print(f"  retention messages -> {OUT.relative_to(C.ROOT)} ({len(out)} drafted: {llm} by LLM, {len(out) - llm} by template; "
          f"guardrails passed {out.guardrail_pass.mean():.0%})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--template-only", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and redraft everything")
    a = ap.parse_args()
    run(n=a.n, template_only=a.template_only, refresh=a.refresh)
