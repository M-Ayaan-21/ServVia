"""
ServVia 3.0 — Multi-Agent Verification Graph
==============================================

LangGraph StateGraph implementing a Proposer → Critic → Fallback
circuit breaker for clinical safety verification.

Architecture:
    ┌──────────┐     ┌──────────┐
    │ Proposer │────▶│  Critic  │
    └──────────┘     └──────────┘
         ▲                │
         │          ┌─────┴─────┐
         │          │           │
         │     approved?   rejected?
         │          │           │
         │          ▼           ▼
         │        [END]   revision_count < 1?
         │                  │          │
         │                 yes         no
         │                  │          │
         └──────────────────┘          ▼
                              ┌──────────────┐
                              │   Fallback   │
                              │ (hardcoded)  │
                              └──────────────┘

Author: ServVia Engineering
Version: 3.0.0
"""

import json
import logging
import asyncio
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agents.prompts import PROPOSER_PROMPT, CRITIC_PROMPT, FALLBACK_RESPONSE

logger = logging.getLogger("ServVia.MultiAgent")


# ═══════════════════════════════════════════════════════════════════════════
# LEAN GRAPH STATE — Minimal payload to save tokens
# ═══════════════════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    user_symptoms: str       # Original user query
    draft_response: str      # Proposer's current output
    critic_feedback: str     # Critic's JSON feedback (for revision)
    revision_count: int      # Circuit breaker counter
    rag_context: str         # RAG chunks (Proposer only — NOT sent to Critic)
    bio_context: str         # Chronobiology advisory string


# ═══════════════════════════════════════════════════════════════════════════
# LLM CALL HELPER — Reuses existing OpenAI service
# ═══════════════════════════════════════════════════════════════════════════

async def _call_llm(prompt: str, temperature: float = 0.3) -> str:
    """
    Call OpenAI using the existing make_openai_request infrastructure.
    Returns the response text or empty string on failure.
    """
    from legacy_agriculture.rag_service.openai_service import make_openai_request

    try:
        response, exception, retries = await make_openai_request(
            prompt, temperature=temperature
        )
        if response and response.choices:
            return response.choices[0].message.content.strip()
        logger.error(f"LLM call failed: {exception}")
        return ""
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# NODE: PROPOSER — Clinical triage assistant
# ═══════════════════════════════════════════════════════════════════════════

async def proposer_node(state: AgentState) -> dict:
    """
    Generate a clinically safe response. On revision, incorporates
    the Critic's feedback to fix issues.
    """
    feedback_section = ""
    if state.get("critic_feedback"):
        feedback_section = (
            f"\n\n=== REVISION REQUIRED ===\n"
            f"Your previous draft was REJECTED by the medical peer reviewer.\n"
            f"Feedback: {state['critic_feedback']}\n"
            f"Fix the issues above in your revised response. "
            f"Do NOT repeat the same mistakes.\n"
        )

    prompt = PROPOSER_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        rag_context=state.get("rag_context", "No additional context available."),
        bio_context=state.get("bio_context", "No chronobiology context."),
        critic_feedback=feedback_section,
    )

    revision = state.get("revision_count", 0)
    logger.info(
        f"📝 Proposer node | Revision #{revision} | "
        f"Query: {state['user_symptoms'][:80]}..."
    )

    draft = await _call_llm(prompt, temperature=0.3)

    if not draft:
        logger.warning("Proposer returned empty — using fallback")
        draft = FALLBACK_RESPONSE

    return {"draft_response": draft}


# ═══════════════════════════════════════════════════════════════════════════
# NODE: CRITIC — Medical peer reviewer
# ═══════════════════════════════════════════════════════════════════════════

async def critic_node(state: AgentState) -> dict:
    """
    Review the Proposer's draft for clinical safety.
    Only receives symptoms + draft (NOT the RAG chunks) to save tokens.
    Outputs minimal JSON: {"is_approved": bool, "feedback": str}
    """
    prompt = CRITIC_PROMPT.format(
        user_symptoms=state["user_symptoms"],
        draft_response=state["draft_response"],
    )

    logger.info("🔍 Critic node | Reviewing draft for clinical safety...")

    raw_output = await _call_llm(prompt, temperature=0.0)

    # Parse the Critic's JSON output
    try:
        # Strip any markdown fences the LLM might add
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
        # If we can't parse, treat as rejection with the raw text as feedback
        is_approved = False
        feedback = raw_output[:300] if raw_output else "Critic output was unparseable"

    logger.info(f"🔍 Critic verdict: approved={is_approved} | {feedback[:100]}")

    # Encode the verdict back into the state
    critic_result = json.dumps({"is_approved": is_approved, "feedback": feedback})

    return {
        "critic_feedback": critic_result,
        "revision_count": state.get("revision_count", 0) + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE: FALLBACK — Zero-LLM hardcoded safe response
# ═══════════════════════════════════════════════════════════════════════════

async def fallback_node(state: AgentState) -> dict:
    """
    Circuit breaker: after 1 failed revision, bypass LLM entirely
    and return a hardcoded safe response.
    """
    logger.critical(
        f"🚨 FALLBACK TRIGGERED | Revision count: {state['revision_count']} | "
        f"Query: {state['user_symptoms'][:80]}"
    )
    return {"draft_response": FALLBACK_RESPONSE}


# ═══════════════════════════════════════════════════════════════════════════
# CONDITIONAL ROUTING — Circuit breaker logic
# ═══════════════════════════════════════════════════════════════════════════

def route_after_critic(state: AgentState) -> str:
    """
    Route based on Critic's verdict and revision count:
      - approved → END
      - rejected + revision_count < 2 → back to Proposer (revision)
      - rejected + revision_count >= 2 → Fallback
    """
    try:
        verdict = json.loads(state.get("critic_feedback", "{}"))
        is_approved = verdict.get("is_approved", False)
    except (json.JSONDecodeError, TypeError):
        is_approved = False

    if is_approved:
        logger.info("✅ Critic APPROVED — routing to final output")
        return "approved"

    revision_count = state.get("revision_count", 0)

    if revision_count < 2:  # Allow 1 revision (count increments in critic: 1 after first, 2 after second)
        logger.info(f"🔄 Critic REJECTED — routing to Proposer for revision #{revision_count}")
        return "revise"
    else:
        logger.warning(f"⛔ Critic REJECTED x{revision_count} — routing to FALLBACK")
        return "fallback"


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH BUILDER — Compile the LangGraph workflow
# ═══════════════════════════════════════════════════════════════════════════

def build_verification_graph() -> StateGraph:
    """
    Build and compile the multi-agent verification graph.

    Flow:
        proposer → critic → [approved: END | revise: proposer | fallback: fallback_node → END]

    Returns:
        Compiled LangGraph runnable.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("proposer", proposer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("fallback", fallback_node)

    # Set entry point
    graph.set_entry_point("proposer")

    # Proposer → Critic (always)
    graph.add_edge("proposer", "critic")

    # Critic → conditional routing
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "approved": END,
            "revise": "proposer",
            "fallback": "fallback",
        },
    )

    # Fallback → END (always)
    graph.add_edge("fallback", END)

    compiled = graph.compile()
    logger.info("🏗️ Multi-Agent Verification Graph compiled successfully")
    return compiled


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON — Compile once, reuse across requests
# ═══════════════════════════════════════════════════════════════════════════

_compiled_graph = None


def get_verification_graph():
    """Get or create the compiled verification graph singleton."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_verification_graph()
    return _compiled_graph


async def run_verification_pipeline(
    user_symptoms: str,
    rag_context: str = "",
    bio_context: str = "",
) -> str:
    """
    Execute the full Proposer → Critic → Fallback pipeline.

    Args:
        user_symptoms: The user's original symptom query.
        rag_context: RAG-retrieved knowledge chunks.
        bio_context: Chronobiology advisory string.

    Returns:
        The final verified (or fallback) response string.
    """
    graph = get_verification_graph()

    initial_state: AgentState = {
        "user_symptoms": user_symptoms,
        "draft_response": "",
        "critic_feedback": "",
        "revision_count": 0,
        "rag_context": rag_context[:3000],  # Cap to save tokens
        "bio_context": bio_context,
    }

    logger.info(f"🚀 Starting Multi-Agent Pipeline | Query: {user_symptoms[:80]}...")

    # Run the graph asynchronously
    final_state = await graph.ainvoke(initial_state)

    pipeline_path = "proposer→critic"
    revision_count = final_state.get("revision_count", 0)
    if revision_count > 1:
        pipeline_path += f"→revision({revision_count - 1})"

    # Check if fallback was triggered
    if final_state.get("draft_response") == FALLBACK_RESPONSE:
        pipeline_path += "→FALLBACK"

    logger.info(f"✅ Multi-Agent Pipeline complete | Path: {pipeline_path}")

    return final_state.get("draft_response", FALLBACK_RESPONSE)
