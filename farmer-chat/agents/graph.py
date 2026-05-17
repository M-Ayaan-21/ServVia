"""
ServVia 4.0 — Multi-Agent Verification Graph
==============================================

LangGraph StateGraph:
  Reasoner -> Proposer -> Critic -> Safety Validator -> [Fallback]

New in v4.0:
  - reasoner_node: structured pre-analysis using deepseek-r1-distill-llama-70b
    (chain-of-thought medical reasoning — differential, evidence, personalisation)
    injects reasoning_context into Proposer for higher-quality responses.
  - Proposer no longer fast-paths the RAG draft — always regenerates using
    the richer reasoning context.
  - Improved prompts throughout.

Flow:
  - First pass: Reasoner -> Proposer -> Critic -> Safety Validator
  - Safety pivot: Proposer (with safety feedback) -> Safety Validator (skip Critic)
  - After 2 failed safety pivots: deterministic safe response (no LLM)
  - Reasoner uses GROQ_REASONER_MODEL (deepseek-r1); Proposer uses GROQ_MODEL (llama-3.3-70b);
    Critic uses GROQ_FAST_MODEL (llama-3.1-8b-instant).

Author: ServVia Engineering
Version: 4.0.0
"""

import json
import logging
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.prompts import PROPOSER_PROMPT, CRITIC_PROMPT, FALLBACK_PROMPT, MEDICAL_REASONING_PROMPT

# Herb alias map so safety feedback explicitly covers all name variants
_HERB_ALIASES_FOR_FEEDBACK = {
    "ginger": "ginger (including ginger root, ginger tea, ginger powder, adrak, zingiber)",
    "turmeric": "turmeric (including curcumin, haldi, curcuma)",
    "garlic": "garlic (including garlic extract, allicin, lahsun)",
    "st. john's wort": "St. John's Wort (including hypericum, SJW)",
    "ginkgo": "ginkgo (including ginkgo biloba)",
    "valerian": "valerian (including valerian root)",
    "kava": "kava (including kava kava, piper methysticum)",
    "ashwagandha": "ashwagandha (including withania, winter cherry)",
    "licorice": "licorice (including mulethi, glycyrrhiza)",
    "echinacea": "echinacea (including coneflower)",
    "grapefruit": "grapefruit (including grapefruit juice)",
}

logger = logging.getLogger("ServVia.MultiAgent")


# ===========================================================================
# LEAN GRAPH STATE
# ===========================================================================

class AgentState(TypedDict):
    user_symptoms: str           # Original user query
    draft_response: str          # Proposer's current output
    critic_feedback: str         # Critic's JSON feedback (for revision)
    revision_count: int          # Critic circuit-breaker counter
    safety_revision_count: int   # Safety-validator circuit-breaker (separate budget)
    rag_context: str             # RAG knowledge context shown to Proposer
    bio_context: str             # Chronobiology advisory string
    is_fallback: bool            # Whether fallback was triggered
    fallback_reason: str         # "emergency" | "drug_interaction" | "critic_loop"
    medical_profile: dict        # User's medical profile for safety checks
    safety_feedback: str         # Deterministic safety feedback injected into Proposer
    reasoning_context: str       # Structured medical reasoning from reasoner_node


# ===========================================================================
# LLM CALL HELPERS
# ===========================================================================

async def _call_llm(prompt: str, temperature: float = 0.3, fast: bool = False) -> str:
    """
    Call Groq via the existing make_openai_request infrastructure.
    fast=True  → GROQ_FAST_MODEL  (llama-3.1-8b-instant, for Critic)
    fast=False → GROQ_MODEL       (llama-3.3-70b-versatile, for Proposer/Fallback)
    """
    from legacy_agriculture.rag_service.openai_service import make_openai_request
    from django_core.config import Config

    model = Config.GROQ_FAST_MODEL if fast else Config.GROQ_MODEL
    try:
        response, exception, retries = await make_openai_request(
            prompt, model=model, temperature=temperature
        )
        if response and response.choices:
            return response.choices[0].message.content.strip()
        logger.error(f"LLM call failed: {exception}")
        return ""
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return ""


async def _call_reasoner(prompt: str, temperature: float = 0.6) -> str:
    """
    Call the reasoning model (deepseek-r1-distill-llama-70b by default).
    Strips <think>...</think> chain-of-thought traces — returns clean output only.
    Gracefully falls back to GROQ_MODEL if the reasoning model is unavailable.
    """
    from legacy_agriculture.rag_service.openai_service import make_reasoner_request
    try:
        return await make_reasoner_request(prompt, temperature=temperature)
    except Exception as e:
        logger.warning(f"Reasoner call failed: {e} — falling back to _call_llm")
        return await _call_llm(prompt, temperature=temperature, fast=False)


# ===========================================================================
# DETERMINISTIC SAFE-ALTERNATIVE RESPONSE BUILDER
# Used when the LLM persistently re-suggests blocked herbs.
# No LLM call — fully deterministic.
# ===========================================================================

def _build_deterministic_safe_response(
    blocked_herbs: list,
    active_meds: str,
) -> str:
    """
    Return a complete, warm, medication-safe response without calling the LLM.
    Invoked after 2 consecutive failed safety pivots.
    """
    herbs_str = " and ".join(blocked_herbs) if blocked_herbs else "certain herbal remedies"
    plural = len(blocked_herbs) > 1
    interact_verb = "interact" if plural else "interacts"

    return f"""I'm sorry to hear you're not feeling well. Because you're taking {active_meds}, I need to avoid recommending {herbs_str} — {('they' if plural else 'it')} {interact_verb} with your medication and can increase certain health risks. Here are safe, effective alternatives that work just as well:

**Remedy 1: Steam Inhalation**
- **Ingredients:** A bowl of boiling water (plain — no herbs needed)
- **Preparation:** Pour hot water into a heatproof bowl on a stable surface.
- **How to use:** Drape a towel over your head, lean over the bowl, and inhale slowly for 5–10 minutes.
- **Frequency:** 2–3 times daily.
- **Why it helps:** Opens nasal passages, loosens congestion, and eases breathing discomfort.

**Remedy 2: Warm Salt Water Gargle**
- **Ingredients:** 1/4 teaspoon table salt dissolved in 1 cup warm water
- **Preparation:** Stir until fully dissolved — it should taste mildly salty, not harsh.
- **How to use:** Gargle for 30 seconds, then spit. Repeat 2–3 times per session.
- **Frequency:** Every 2–3 hours while symptomatic.
- **Why it helps:** Reduces throat inflammation and flushes out irritants safely.

**Remedy 3: Honey and Lemon in Warm Water**
- **Ingredients:** 1 tablespoon raw honey, juice of half a lemon, 1 cup warm (not boiling) water
- **Preparation:** Stir honey and lemon into warm water until dissolved.
- **How to use:** Sip slowly while warm.
- **Frequency:** 2–3 cups daily.
- **Why it helps:** Honey has natural antimicrobial properties; lemon provides vitamin C support — both safe alongside {active_meds}.

**Remedy 4: Warm Compress**
- **Ingredients:** Clean cloth, warm water
- **Preparation:** Soak cloth in warm water and wring out excess.
- **How to use:** Apply to forehead, chest, or the back of the neck for 10–15 minutes.
- **Frequency:** As needed for comfort.
- **Why it helps:** Relieves body aches and reduces fever discomfort without any medication interaction.

**Remedy 5: Rest and Warm Fluids**
- **What to drink:** Warm water, clear broth, peppermint tea, or chamomile tea — all safe alongside {active_meds}.
- **How to use:** Aim for 8–10 glasses of warm fluids throughout the day. Rest in a slightly elevated position to ease breathing.
- **Frequency:** Throughout the day.
- **Why it helps:** Prevents dehydration, supports your immune system, and speeds recovery.

**Safety Notes:**
- Raw honey, lemon, peppermint, and chamomile are generally safe with {active_meds} — but confirm with your pharmacist if you are unsure.
- Avoid adding grapefruit juice to your fluids, as it can interact with some medications.

**When to see a doctor:**
- Fever above 103°F (39.4°C) or lasting more than 3 days
- Fever accompanied by stiff neck, severe headache, or a rash — seek emergency care immediately
- Symptoms worsening despite these measures
- Any difficulty breathing or chest pain

*I recommend asking your doctor or pharmacist: "Which home remedies are safe alongside {active_meds}?" for personalized guidance.*"""


# ===========================================================================
# NODE: REASONER -- Structured pre-analysis using deepseek-r1
# Runs once at the start of every pipeline (revision_count == 0).
# Outputs structured JSON with differential, evidence, personalisation.
# The Proposer uses this context to generate a more accurate response.
# ===========================================================================

async def reasoner_node(state: AgentState) -> dict:
    """
    Pre-Proposer structured clinical reasoning.

    Uses deepseek-r1-distill-llama-70b which does native chain-of-thought
    reasoning (emitted as <think> tags, stripped automatically).

    Produces reasoning_context JSON injected into the Proposer prompt.
    If USE_REASONER=false or the model fails, returns an empty string
    so the Proposer can still run normally without it.
    """
    from django_core.config import Config

    if not getattr(Config, "USE_REASONER", True):
        logger.info("Reasoner disabled (USE_REASONER=false) — skipping")
        return {"reasoning_context": ""}

    profile_data = state.get("medical_profile", {})
    meds_list = profile_data.get("current_medications", [])
    allergies_list = profile_data.get("allergies", [])
    meds_str = ", ".join(str(m) for m in meds_list if m) or "none reported"
    allergies_str = ", ".join(str(a) for a in allergies_list if a) or "none reported"
    medical_profile_str = f"Current medications: {meds_str}\nAllergies: {allergies_str}"

    prompt = MEDICAL_REASONING_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        medical_profile_str=medical_profile_str,
        rag_context=state.get("rag_context", "No knowledge context available."),
        bio_context=state.get("bio_context", "No chronobiology context."),
    )

    logger.info(f"Reasoner node | Query: {state['user_symptoms'][:80]}...")

    raw = await _call_reasoner(prompt, temperature=0.6)

    # Validate it's parseable JSON — if not, still use it as free-text context
    reasoning_str = raw or ""
    try:
        cleaned = reasoning_str.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        json.loads(cleaned)
        reasoning_str = cleaned
        logger.info(f"Reasoner produced valid JSON | length={len(reasoning_str)}")
    except (json.JSONDecodeError, AttributeError):
        # Use raw text as context — still better than nothing
        reasoning_str = reasoning_str[:1200] if reasoning_str else ""
        logger.warning("Reasoner JSON parse failed — using raw text as context")

    return {"reasoning_context": reasoning_str}


# ===========================================================================
# NODE: PROPOSER -- Clinical triage assistant
# ===========================================================================

async def proposer_node(state: AgentState) -> dict:
    """
    Generate a clinically safe, evidence-based response.

    On first pass: uses reasoning_context (from reasoner_node) to produce
    a higher-quality response informed by structured clinical analysis.

    On revision: incorporates Critic or Safety Validator feedback.
    No fast-path bypass — always runs the LLM for quality.
    """
    revision = state.get("revision_count", 0)
    safety_revision = state.get("safety_revision_count", 0)

    # -- Build reasoning context section ------------------------------------
    reasoning_raw = state.get("reasoning_context", "")
    if reasoning_raw:
        # Try to pretty-print JSON for readability in the prompt
        try:
            parsed = json.loads(reasoning_raw)
            reasoning_display = json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, ValueError):
            reasoning_display = reasoning_raw

        reasoning_section = (
            "=== CLINICAL REASONING ANALYSIS (from pre-analysis step) ===\n"
            "Use this structured analysis to guide your response — it contains differential "
            "diagnosis, evidence review, personalisation notes, and chronobiology guidance.\n\n"
            f"{reasoning_display}\n"
            "=== END REASONING CONTEXT ==="
        )
    else:
        reasoning_section = ""

    # -- Build feedback section ----------------------------------------------
    feedback_section = ""
    if state.get("safety_feedback"):
        feedback_section = (
            f"\n\n=== REVISION REQUIRED (SAFETY BLOCK) ===\n"
            f"{state['safety_feedback']}\n"
        )
    elif state.get("critic_feedback"):
        feedback_section = (
            f"\n\n=== REVISION REQUIRED ===\n"
            f"Your previous draft was REJECTED by the medical peer reviewer.\n"
            f"Feedback: {state['critic_feedback']}\n"
            f"Fix the issues above. Do NOT repeat the same mistakes.\n"
        )

    prompt = PROPOSER_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        rag_context=state.get("rag_context", "No additional context available."),
        bio_context=state.get("bio_context", "No chronobiology context."),
        reasoning_context=reasoning_section,
        critic_feedback=feedback_section,
    )

    logger.info(
        f"Proposer node | Critic rev #{revision} | Safety rev #{safety_revision} | "
        f"has_reasoning={'yes' if reasoning_raw else 'no'} | "
        f"Query: {state['user_symptoms'][:80]}..."
    )

    draft = await _call_llm(prompt, temperature=0.3, fast=False)

    if not draft:
        logger.warning("Proposer returned empty -- will route to fallback")
        draft = ""

    return {
        "draft_response": draft,
        "is_fallback": False,
        "safety_feedback": "",
        "critic_feedback": "",
    }


# ===========================================================================
# NODE: CRITIC -- Medical peer reviewer (uses fast model)
# ===========================================================================

async def critic_node(state: AgentState) -> dict:
    """
    Review the Proposer's draft for clinical safety.
    Uses the fast model — only needs to output a small JSON verdict.
    """
    prompt = CRITIC_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        draft_response=state["draft_response"],
    )

    logger.info("Critic node | Reviewing draft (fast model)...")

    raw_output = await _call_llm(prompt, temperature=0.0, fast=True)

    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        verdict = json.loads(cleaned)
        is_approved = verdict.get("is_approved", False)
        feedback = verdict.get("feedback", "No feedback provided")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Critic JSON parse failed: {e} | Raw: {raw_output[:200]}")
        is_approved = False
        feedback = raw_output[:300] if raw_output else "Critic output was unparseable"

    logger.info(f"Critic verdict: approved={is_approved} | {feedback[:100]}")

    new_revision_count = state.get("revision_count", 0) + 1
    fallback_reason = state.get("fallback_reason", "")
    if not is_approved and new_revision_count >= 3 and not fallback_reason:
        fallback_reason = "critic_loop"

    return {
        "critic_feedback": json.dumps({"is_approved": is_approved, "feedback": feedback}),
        "revision_count": new_revision_count,
        "fallback_reason": fallback_reason,
    }


# ===========================================================================
# NODE: SAFETY VALIDATOR -- Deterministic Contraindication Check
# ===========================================================================

async def safety_validator_node(state: AgentState) -> dict:
    """
    Scan the approved draft for ALL contraindicated herbs in a single pass.

    Attempt 1 (safety_revision_count becomes 1): send explicit forbidden-list
      message back to Proposer.
    Attempt 2 (safety_revision_count becomes 2): LLM is not complying.
      Build a deterministic safe-alternative response and route to END.
    """
    from neurosymbolic.temporal_validator import TemporalSafetyValidator
    from core.models import UserMedicalProfile, RemedyProposal, MedicationRecord
    from datetime import datetime, timezone, timedelta
    from api.views import _extract_herbs_from_response

    validator = TemporalSafetyValidator()
    profile_data = state.get("medical_profile", {})

    med_records = []
    for med_name in profile_data.get("current_medications", []):
        if isinstance(med_name, str) and med_name.strip():
            med_records.append(
                MedicationRecord(
                    drug_name=med_name.strip(),
                    start_date=datetime.now(timezone.utc) - timedelta(days=90),
                    end_date=None,
                )
            )

    medical_profile = UserMedicalProfile(
        user_id="pipeline_user",
        allergies=profile_data.get("allergies", []),
        current_medications=med_records,
        symptom_onset_hours=0,
    )

    draft = state["draft_response"]
    proposed_herbs = _extract_herbs_from_response(draft)

    # -- Collect ALL violations in one pass ----------------------------------
    blocked = []
    for herb_name in proposed_herbs:
        proposal = RemedyProposal(
            herb_or_remedy_name=herb_name,
            intended_effect="LLM-recommended remedy",
        )
        result = validator.validate_remedy(medical_profile, proposal)
        if not result.is_safe:
            blocked.append((herb_name, result.reason))
            logger.warning(f"Safety block: '{herb_name}' -- {result.reason[:120]}")

    if not blocked:
        logger.info("Safety Validator: all clear")
        return {"safety_feedback": ""}

    new_safety_revision_count = state.get("safety_revision_count", 0) + 1

    logger.warning(
        f"Safety Validator blocked {len(blocked)} herb(s) "
        f"(attempt #{new_safety_revision_count}): {[h for h, _ in blocked]}"
    )

    active_meds_list = [
        m.strip() for m in profile_data.get("current_medications", []) if m.strip()
    ]
    active_meds = ", ".join(active_meds_list) or "your current medications"

    # -- Second consecutive block: LLM is not complying — go deterministic --
    if new_safety_revision_count >= 2:
        logger.warning(
            "Safety Validator: 2 consecutive blocks -- building deterministic "
            "safe response and routing to END."
        )
        blocked_herb_names = [h for h, _ in blocked]
        canned_response = _build_deterministic_safe_response(
            blocked_herbs=blocked_herb_names,
            active_meds=active_meds,
        )
        return {
            "draft_response": canned_response,
            "safety_feedback": "",          # clears block -> route_after_safety -> "safe" -> END
            "safety_revision_count": new_safety_revision_count,
            "is_fallback": False,
            "fallback_reason": "drug_interaction",
        }

    # -- First block: build explicit forbidden-list for the LLM --------------
    # Carry forward any herbs blocked in previous rounds
    prev_feedback = state.get("safety_feedback", "")
    prev_blocked_lines = []
    if "PERMANENTLY FORBIDDEN HERBS" in prev_feedback:
        start = prev_feedback.find("PERMANENTLY FORBIDDEN HERBS")
        end = prev_feedback.find("\n\nREASONS:", start)
        if end == -1:
            end = prev_feedback.find("\n\n", start + 10)
        if end != -1:
            prev_blocked_lines = [
                ln.strip("- ").strip()
                for ln in prev_feedback[start:end].splitlines()
                if ln.strip().startswith("-")
            ]

    all_forbidden_display = list(prev_blocked_lines)
    for herb_name, _ in blocked:
        display = _HERB_ALIASES_FOR_FEEDBACK.get(herb_name, herb_name)
        if display not in all_forbidden_display:
            all_forbidden_display.append(display)

    reasons_section = "\n".join(
        f"  - {herb}: {reason}" for herb, reason in blocked
    )
    forbidden_section = "\n".join(f"  - {h}" for h in all_forbidden_display)
    blocked_names_str = ", ".join(h for h, _ in blocked)

    combined_msg = f"""
==========================================================
           SAFETY SYSTEM OVERRIDE -- DRAFT BLOCKED
==========================================================

PERMANENTLY FORBIDDEN HERBS (DO NOT MENTION IN ANY FORM):
{forbidden_section}

REASONS FOR THIS BLOCK:
{reasons_section}

USER CONTEXT: The user is taking {active_meds}. These medications interact with the herbs listed above and could cause serious harm (e.g., increased bleeding risk, dangerous drug levels).

MANDATORY PIVOT INSTRUCTIONS -- FOLLOW EXACTLY:

STEP 1 -- ACKNOWLEDGE (do this first, briefly):
  Tell the user warmly: "Because you're taking [medication], I need to avoid [blocked herb(s)] as they can interact with your medication. Here are some safe alternatives instead."

STEP 2 -- REPLACE WITH SAFE ALTERNATIVES (choose 4-5):
  - Warm salt water gargle
  - Steam inhalation (plain hot water -- no herbs needed)
  - Honey and lemon in warm water
  - Saline nasal rinse
  - Warm peppermint or chamomile tea (safe with most medications)
  - Warm broth or chicken soup
  - Warm compress on forehead or chest
  - Cool mist humidifier
  - Rest in elevated position
  - Increased warm water intake

STEP 3 -- CLOSE WITH DOCTOR NOTE:
  "I recommend asking your doctor or pharmacist which herbal remedies are safe alongside your current medications."

STEP 4 -- DO NOT:
  - Do NOT suggest {blocked_names_str} in any form, variation, or synonym.
  - Do NOT give a vague "sorry, can't help" response. You MUST provide the safe alternatives.
""".strip()

    return {
        "safety_feedback": combined_msg,
        "safety_revision_count": new_safety_revision_count,
        "fallback_reason": "drug_interaction",
    }


# ===========================================================================
# NODE: FALLBACK -- LLM-powered context-aware response
# ===========================================================================

async def fallback_node(state: AgentState) -> dict:
    """
    Circuit breaker: triggered only by critic loop exhaustion or genuine
    emergency. Drug-interaction exhaustion is handled deterministically in
    safety_validator_node and never reaches here.
    """
    fallback_reason = state.get("fallback_reason") or ""
    if not fallback_reason:
        if state.get("safety_revision_count", 0) >= 2:
            fallback_reason = "drug_interaction"
        elif state.get("revision_count", 0) >= 3:
            fallback_reason = "critic_loop"
        else:
            fallback_reason = "emergency"

    logger.critical(
        f"FALLBACK TRIGGERED | Reason: {fallback_reason} | "
        f"Critic revisions: {state.get('revision_count', 0)} | "
        f"Safety revisions: {state.get('safety_revision_count', 0)} | "
        f"Query: {state['user_symptoms'][:80]}"
    )

    prompt = FALLBACK_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        fallback_reason=fallback_reason,
    )

    raw_output = await _call_llm(prompt, temperature=0.4, fast=False)

    if not raw_output:
        fallback_json = json.dumps({
            "is_emergency_block": fallback_reason == "emergency",
            "has_remedies": False,
            "response_text": (
                "I was unable to generate a safe response. "
                "Please seek professional medical advice."
            ),
        })
        return {"draft_response": fallback_json, "is_fallback": True}

    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
        json.loads(cleaned)
        return {"draft_response": cleaned, "is_fallback": True}
    except Exception as e:
        logger.error(f"Fallback JSON parse failed: {e}")
        fallback_json = json.dumps({
            "is_emergency_block": fallback_reason == "emergency",
            "has_remedies": False,
            "response_text": raw_output,
        })
        return {"draft_response": fallback_json, "is_fallback": True}


# ===========================================================================
# CONDITIONAL ROUTING
# ===========================================================================

def route_after_proposer(state: AgentState) -> str:
    """
    After a safety pivot the Critic already approved the structure — only safety
    needs to re-check. Skip the Critic to save one LLM call per pivot.
    """
    if state.get("safety_revision_count", 0) > 0:
        logger.info("Proposer -> Safety Validator direct (skipping Critic on safety pivot)")
        return "safety_validator"
    return "critic"


def route_after_critic(state: AgentState) -> str:
    """
    - approved -> safety_validator
    - rejected + revision_count < 3 -> proposer
    - rejected + revision_count >= 3 -> fallback
    """
    try:
        verdict = json.loads(state.get("critic_feedback", "{}"))
        is_approved = verdict.get("is_approved", False)
    except (json.JSONDecodeError, TypeError):
        is_approved = False

    if is_approved:
        logger.info("Critic APPROVED -- routing to Safety Validator")
        return "approved"

    revision_count = state.get("revision_count", 0)
    if revision_count < 3:
        logger.info(f"Critic REJECTED -- routing to Proposer for revision #{revision_count}")
        return "revise"
    else:
        logger.warning(f"Critic REJECTED x{revision_count} -- routing to FALLBACK")
        return "fallback"


def route_after_safety(state: AgentState) -> str:
    """
    - No safety_feedback -> END
    - Has safety_feedback + safety_revision_count < 2 -> proposer (pivot)
    - Has safety_feedback + safety_revision_count >= 2 -> fallback (last resort)
    """
    if not state.get("safety_feedback"):
        logger.info("Safety Validator CLEAR -- routing to END")
        return "safe"

    safety_revision_count = state.get("safety_revision_count", 0)
    if safety_revision_count < 2:
        logger.warning(f"Safety BLOCKED -- routing to Proposer for pivot #{safety_revision_count}")
        return "revise"
    else:
        logger.warning(f"Safety BLOCKED x{safety_revision_count} -- routing to FALLBACK")
        return "fallback"


# ===========================================================================
# GRAPH BUILDER
# ===========================================================================

def build_verification_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("reasoner", reasoner_node)
    graph.add_node("proposer", proposer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("safety_validator", safety_validator_node)
    graph.add_node("fallback", fallback_node)

    # Entry: Reasoner (structured analysis) -> Proposer (response generation)
    graph.set_entry_point("reasoner")
    graph.add_edge("reasoner", "proposer")

    # Proposer -> Critic OR Safety Validator (skip Critic on safety pivots)
    graph.add_conditional_edges(
        "proposer",
        route_after_proposer,
        {"critic": "critic", "safety_validator": "safety_validator"},
    )

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"approved": "safety_validator", "revise": "proposer", "fallback": "fallback"},
    )

    graph.add_conditional_edges(
        "safety_validator",
        route_after_safety,
        {"safe": END, "revise": "proposer", "fallback": "fallback"},
    )

    graph.add_edge("fallback", END)

    compiled = graph.compile()
    logger.info("Multi-Agent Verification Graph v4.0 compiled: reasoner->proposer->critic->safety->fallback")
    return compiled


# ===========================================================================
# SINGLETON
# ===========================================================================

_compiled_graph = None


def get_verification_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_verification_graph()
    return _compiled_graph


async def run_verification_pipeline(
    user_symptoms: str,
    rag_context: str = "",
    bio_context: str = "",
    medical_profile: dict = None,
    initial_draft: str = "",
) -> dict:
    """
    Execute the full Proposer -> Critic -> Safety Validator -> Fallback pipeline.

    Args:
        user_symptoms:  The user's original symptom query.
        rag_context:    RAG-retrieved knowledge chunks shown to Proposer.
        bio_context:    Chronobiology advisory string.
        medical_profile: User profile containing meds and allergies.
        initial_draft:  Pre-generated RAG draft. If provided, the Proposer skips
                        its first LLM call and uses this as the starting draft,
                        saving one API call per request.

    Returns:
        dict with keys: is_emergency_block, has_remedies, response_text.
    """
    graph = get_verification_graph()

    initial_state: AgentState = {
        "user_symptoms": user_symptoms,
        "draft_response": initial_draft,    # RAG draft carried as context reference
        "critic_feedback": "",
        "revision_count": 0,
        "safety_revision_count": 0,
        "rag_context": str(rag_context)[:3500],
        "bio_context": str(bio_context),
        "is_fallback": False,
        "fallback_reason": "",
        "medical_profile": medical_profile or {},
        "safety_feedback": "",
        "reasoning_context": "",            # populated by reasoner_node
    }

    logger.info(
        f"Starting Multi-Agent Pipeline | "
        f"pre-seeded={'yes' if initial_draft else 'no'} | "
        f"Query: {user_symptoms[:80]}..."
    )

    final_state = await graph.ainvoke(initial_state)

    revision_count = final_state.get("revision_count", 0)
    safety_revision_count = final_state.get("safety_revision_count", 0)
    fallback_reason = final_state.get("fallback_reason", "")

    path_parts = ["proposer->critic"]
    if revision_count > 1:
        path_parts.append(f"revision_loop({revision_count - 1})")
    if safety_revision_count > 0:
        path_parts.append(f"safety_pivot({safety_revision_count})")
    if final_state.get("is_fallback"):
        path_parts.append(f"FALLBACK({fallback_reason})")
    pipeline_path = "->".join(path_parts)

    logger.info(f"Multi-Agent Pipeline complete | Path: {pipeline_path}")

    if final_state.get("is_fallback", False):
        try:
            return json.loads(final_state.get("draft_response", "{}"))
        except Exception:
            return {
                "is_emergency_block": fallback_reason == "emergency",
                "has_remedies": False,
                "response_text": final_state.get("draft_response", ""),
            }

    return {
        "is_emergency_block": False,
        "has_remedies": True,
        "response_text": final_state.get("draft_response", ""),
    }
