"""
ServVia 3.0 — Unified Healthcare Pipeline View
================================================

This module replaces the legacy farmer-chat routing with the strict
ServVia Neurosymbolic Pipeline. Every chat request flows through:

    Step A: Emergency Detection (hardcoded safety layer — no LLM)
    Step B: Chronobiology Context  (passive biological state inference)
    Step C: RAG Generation         (LLM-powered response via legacy pipeline)
    Step D: Safety Validation      (temporal neurosymbolic drug-herb check)
    Step E: Final Output           (safe response or contraindication block)

Author: ServVia Engineering
Version: 3.0.0
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

# ─── Legacy infrastructure (auth, translate, RAG) ───────────────────────
from legacy_agriculture.api.utils import (
    authenticate_user_based_on_email,
    handle_input_query,
    mask_email,
    preprocess_user_data,
)
from legacy_agriculture.common.utils import (
    get_user_chat_history,
    save_message_obj,
    postprocess_and_translate_query_response,
)
from legacy_agriculture.language_service.translation import (
    detect_language_and_translate_to_english,
)
from legacy_agriculture.rag_service.execute_rag import execute_rag_pipeline

# ─── New ServVia modules ────────────────────────────────────────────────
from core.models import (
    MedicationRecord,
    RemedyProposal,
    UserMedicalProfile,
    ValidationResult,
)
from chronobiology.inference import ChronobiologyEngine
from neurosymbolic.temporal_validator import TemporalSafetyValidator
from legacy_agriculture.rag_service.execute_rag import EmergencySystem

# Graph RAG — Outcome Adaptive Ranking
try:
    from graph_rag.ranker import OutcomeAdaptiveRanker
    _adaptive_ranker = OutcomeAdaptiveRanker()
    GRAPH_RAG_AVAILABLE = True
except ImportError:
    GRAPH_RAG_AVAILABLE = False

# Trust Engine for scientific validation
try:
    from servvia2.trust_engine.engine import get_trust_engine
    TRUST_ENGINE_AVAILABLE = True
except ImportError:
    TRUST_ENGINE_AVAILABLE = False

# DB-backed Temporal Reasoning Engine (checks real MedicationHistory rows)
try:
    from servvia2.temporal_reasoning.engine import get_temporal_engine
    TEMPORAL_ENGINE_AVAILABLE = True
except ImportError:
    TEMPORAL_ENGINE_AVAILABLE = False

# Multi-Agent LangGraph verification
try:
    from agents.graph import run_verification_pipeline
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-Agent graph not available: {e}")
    MULTI_AGENT_AVAILABLE = False

# Fast Triage Layer — intercepts serious symptom clusters before RAG
try:
    from agents.triage import detect_symptom_cluster, generate_cluster_response
    TRIAGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Triage layer not available: {e}")
    TRIAGE_AVAILABLE = False

logger = logging.getLogger("ServVia.PipelineView")

# ─── Singleton instances (initialized once at import time) ──────────────
_chrono_engine = ChronobiologyEngine()
_safety_validator = TemporalSafetyValidator()


def _get_client_ip(request) -> str:
    """Extract real client IP from request, respecting X-Forwarded-For."""
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "")


# =============================================================================
# HELPER: BUILD MEDICAL PROFILE FROM USER PROFILE DATA
# =============================================================================

def _build_medical_profile(email_id: str, profile_data: dict = None) -> UserMedicalProfile:
    """
    Build a typed Pydantic UserMedicalProfile for the TemporalSafetyValidator.

    Prefers real MedicationHistory DB records (with accurate start/stop dates)
    over the comma-separated UserProfile.current_medications string.
    Falls back to the string list with a 90-day assumed start date only when
    no MedicationHistory rows exist for this user.

    Args:
        email_id: User's email for identification.
        profile_data: Dict from UserProfile model (allergies, medications, etc.)

    Returns:
        UserMedicalProfile: Typed, validated profile.
    """
    allergies = profile_data.get("allergies", []) or [] if profile_data else []

    # ── Prefer real MedicationHistory records (temporal-aware) ──────────────
    med_records = []
    try:
        from user_profile.models import MedicationHistory, UserProfile
        user = UserProfile.objects.get(email=email_id)
        for med in MedicationHistory.objects.filter(user=user):
            start = med.start_date
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            stop = None
            if med.stop_date:
                stop = med.stop_date
                if stop.tzinfo is None:
                    stop = stop.replace(tzinfo=timezone.utc)
            med_records.append(
                MedicationRecord(
                    drug_name=med.medication_name,
                    start_date=start,
                    end_date=stop,
                )
            )
        logger.info(
            f"📅 Loaded {len(med_records)} MedicationHistory records "
            f"(with real dates) for {email_id}"
        )
    except Exception:
        pass  # Fall through to string-based fallback below

    # ── Fallback: parse comma-separated string, assume started 90 days ago ──
    if not med_records and profile_data:
        raw_meds = profile_data.get("current_medications", []) or []
        for med_name in raw_meds:
            if isinstance(med_name, str) and med_name.strip():
                med_records.append(
                    MedicationRecord(
                        drug_name=med_name.strip(),
                        start_date=datetime.now(timezone.utc) - timedelta(days=90),
                        end_date=None,
                    )
                )
        if med_records:
            logger.warning(
                f"⚠️ No MedicationHistory rows found for {email_id}; "
                f"using profile string fallback with assumed 90-day start dates. "
                f"Washout checks will be inaccurate."
            )

    return UserMedicalProfile(
        user_id=email_id,
        allergies=allergies,
        current_medications=med_records,
        symptom_onset_hours=0,
    )


# =============================================================================
# HELPER: EXTRACT HERBS FROM LLM RESPONSE FOR SAFETY VALIDATION
# =============================================================================

# Common herbs the LLM might recommend
_HERB_SCAN_LIST = [
    "ginger", "turmeric", "garlic", "ashwagandha", "chamomile", "valerian",
    "ginseng", "echinacea", "licorice", "ginkgo", "kava", "peppermint",
    "st. john's wort", "st john's wort", "tulsi", "neem", "amla",
    "fenugreek", "cinnamon", "fennel", "cumin", "grapefruit", "honey",
    "aloe vera", "moringa", "triphala", "brahmi", "shatavari",
]

import re

def _extract_herbs_from_response(response_text: str) -> list[str]:
    """
    Scan the LLM response for herb mentions using word-boundary matching.

    Returns:
        List of unique herb names found in the response.
    """
    if not response_text:
        return []

    # Ensure response_text is a string before calling lower()
    if isinstance(response_text, dict):
        response_text = str(response_text)
    elif not isinstance(response_text, str):
        response_text = str(response_text)

    response_lower = response_text.lower()
    found = []
    for herb in _HERB_SCAN_LIST:
        if re.search(r'\b' + re.escape(herb) + r'\b', response_lower):
            found.append(herb)
    return found


# =============================================================================
# HELPER: FORMAT SAFETY BLOCK RESPONSE
# =============================================================================

def _format_safety_block(
    original_response: str,
    validation_result: ValidationResult,
    herb_name: str,
) -> str:
    """
    Format a user-friendly safety block message when a remedy is
    contraindicated. The original LLM response is suppressed.

    Returns:
        Formatted safety message with clear clinical explanation.
    """
    block_response = (
        f"**Safety Alert — Recommendation Blocked**\n\n"
        f"I was about to recommend **{herb_name}**, but our safety system "
        f"has flagged a potential issue:\n\n"
        f"---\n\n"
        f"**{validation_result.reason}**\n\n"
        f"---\n\n"
    )

    if validation_result.washout_days_remaining:
        block_response += (
            f"**Washout Period:** {validation_result.washout_days_remaining} "
            f"day(s) remaining before this remedy is considered safe.\n\n"
        )

    if validation_result.contraindications:
        block_response += "**Specific Contraindications:**\n"
        for ci in validation_result.contraindications:
            block_response += f"  - {ci}\n"
        block_response += "\n"

    block_response += (
        "**What you can do:**\n"
        "  - Consult your doctor before trying herbal remedies with your current medications.\n"
        "  - Ask me about alternative remedies that are safe with your profile.\n"
        "  - Update your medication list in your profile if it's changed.\n\n"
        "*This safety check is powered by ServVia's Neurosymbolic "
        "Pharmacovigilance Engine — a deterministic, evidence-based system.*"
    )

    return block_response


def _build_allergy_block_response(
    allergy_blocked_herbs: list,
    condition: str,
    user_allergies: list,
    user_medications: list,
    user_conditions: list,
    trust_engine,
) -> str:
    """
    Build a hard-block response for allergy violations.
    Looks up safe alternatives from the evidence DB and includes them.
    """
    herb_list = ", ".join(h.title() for h in allergy_blocked_herbs)

    msg = (
        f"**Safety Alert — Recommendation Blocked**\n\n"
        f"The suggested remedy contains **{herb_list}**, which is listed as an allergen "
        f"in your profile. That response has been suppressed.\n\n"
    )

    # Look up safe alternatives from the evidence DB
    try:
        alternatives = trust_engine.find_safe_alternatives(
            condition=condition,
            excluded_herbs=user_allergies,
            user_medications=user_medications,
            user_conditions=user_conditions,
            max_results=3,
        )
    except Exception:
        alternatives = []

    if alternatives:
        msg += f"**Safe alternatives for your condition ({condition.replace('_', ' ')}):**\n\n"
        for alt in alternatives:
            msg += f"- **{alt['herb']}** *(Evidence: {alt['evidence_level']})*\n"
            msg += f"  {alt['summary']}\n"
            if alt['dosing']:
                msg += f"  Suggested use: {alt['dosing']}\n"
            msg += "\n"
    else:
        msg += (
            "No alternative remedies with sufficient evidence were found in our database "
            "for your condition that are safe with your profile. Please consult a doctor.\n\n"
        )

    msg += (
        "**Note:** Always consult a healthcare provider before starting any herbal remedy.\n\n"
        "*This block is enforced by ServVia's Neurosymbolic Safety Engine.*"
    )
    return msg


# =============================================================================
# MAIN PIPELINE VIEW
# =============================================================================

class ServViaChatViewSet(GenericViewSet):
    """
    ServVia 3.0 Chat Pipeline.

    Replaces the legacy ChatAPIViewSet with the strict neurosymbolic
    pipeline: Emergency → Chronobiology → RAG → Safety → Response.

    All endpoints are unauthenticated (matching legacy behavior) with
    email-based identification.
    """

    authentication_classes = []
    permission_classes = []

    @action(detail=False, methods=["post"])
    def get_answer_for_text_query(self, request):
        """
        Primary chat endpoint — receives a text query, returns a
        safety-validated response through the full ServVia pipeline.

        POST /api/chat/get_answer_for_text_query/
        Body: { "email_id": "...", "query": "..." }

        Pipeline:
            A. Emergency Detection
            B. Chronobiology Context
            C. RAG Generation
            D. Safety Validation
            E. Final Output
        """
        email_id = request.data.get("email_id")
        original_query = request.data.get("query")
        response_data = Response(
            {"message": None, "query": original_query, "error": False}
        )

        try:
            # ─── AUTH ────────────────────────────────────────────────
            authenticated_user = authenticate_user_based_on_email(email_id)
            if not authenticated_user:
                response_data.data["message"] = "Invalid Email ID"
                response_data.status_code = status.HTTP_401_UNAUTHORIZED
                return response_data

            if not original_query:
                response_data.data["message"] = "Please submit a query."
                response_data.status_code = status.HTTP_400_BAD_REQUEST
                return response_data

            logger.info(f"🚀 ServVia 3.0 Pipeline | User: {mask_email(email_id)} | Query: {original_query}")

            # ═══════════════════════════════════════════════════════
            # STEP A: EMERGENCY DETECTION (hardcoded, no LLM)
            # ═══════════════════════════════════════════════════════
            emergency_type = EmergencySystem.detect_intent(original_query)
            if emergency_type:
                logger.critical(f"🚨 EMERGENCY DETECTED: {emergency_type}")
                emergency_response = EmergencySystem.get_response(emergency_type)

                response_data.data["message"] = "Emergency response"
                response_data.data["response"] = emergency_response
                response_data.data["pipeline"] = "emergency_intercept"
                response_data.data["emergency_type"] = emergency_type
                return response_data

            # ═══════════════════════════════════════════════════════
            # STEP B: CHRONOBIOLOGY CONTEXT (passive, no LLM)
            # Uses real IP geolocation + live weather data.
            # ═══════════════════════════════════════════════════════
            client_ip = _get_client_ip(request)
            bio_state = _chrono_engine.infer_state_from_request(client_ip)
            logger.info(
                f"🕐 Bio State: phase={bio_state.circadian_phase.value}, "
                f"season={bio_state.seasonal_influence.value}, "
                f"misaligned={bio_state.is_misaligned}, "
                f"weather={bio_state.weather_description}, "
                f"location={bio_state.location_city}"
            )

            # ═══════════════════════════════════════════════════════
            # STEP C: RAG GENERATION (legacy pipeline)
            # ═══════════════════════════════════════════════════════
            # Reuse legacy process_query for translation + RAG + response generation.
            # This internally handles: translate → rephrase → retrieve → generate → translate back.
            import datetime as dt

            user_data, message_obj = preprocess_user_data(
                original_query, email_id, authenticated_user
            )
            user_id = user_data.get("user_id")
            user_name = user_data.get("user_name")
            message_id = user_data.get("message_id")
            chat_history = get_user_chat_history(user_id) if user_id else None

            # Translate input
            query_in_english, input_language_detected = asyncio.run(
                detect_language_and_translate_to_english(original_query)
            )
            logger.info(f"🌐 Language: {input_language_detected}")

            # Load user profile
            user_profile_data = None
            try:
                from user_profile.models import UserProfile
                profile = UserProfile.objects.get(email=email_id)
                user_profile_data = {
                    "allergies": profile.get_allergies_list(),
                    "medical_conditions": profile.get_conditions_list(),
                    "current_medications": profile.get_medications_list(),
                    "first_name": profile.first_name,
                }
                if profile.first_name:
                    user_name = profile.first_name
                logger.info(f"📋 Profile loaded: {profile.first_name}")
            except Exception as e:
                logger.info(f"No profile for {mask_email(email_id)}: {e}")

            # Build bio_context string (used by both fast triage and multi-agent)
            bio_context_str = ""
            if bio_state.advisory:
                bio_context_str = (
                    f"Circadian Phase: {bio_state.circadian_phase.value}, "
                    f"Season: {bio_state.seasonal_influence.value}, "
                    f"Sleep Pressure: {bio_state.sleep_pressure_estimate.value}"
                )

            # ═══════════════════════════════════════════════════════
            # STEP B-KG: GRAPH RAG — OUTCOME ADAPTIVE RANKING
            # Queries IntegrativeKnowledgeGraph for the user's query,
            # applies personalization weights from their remedy outcome
            # history, and produces an evidence-graded context string
            # injected into the multi-agent Proposer.
            # ═══════════════════════════════════════════════════════
            kg_context_str = ""
            if GRAPH_RAG_AVAILABLE and email_id:
                try:
                    allergies_for_kg = (user_profile_data or {}).get("allergies", []) or []
                    meds_for_kg = (user_profile_data or {}).get("current_medications", []) or []
                    _, kg_context_str = _adaptive_ranker.get_top_remedies_for_user(
                        user_email=email_id,
                        condition=query_in_english,
                        exclude_herbs=allergies_for_kg,
                        user_medications=meds_for_kg,
                    )
                    if kg_context_str:
                        logger.info("📊 Graph RAG context generated (personalized)")
                except Exception as e:
                    logger.warning(f"Graph RAG step failed (non-fatal): {e}")

            # ═══════════════════════════════════════════════════════
            # STEP C-FAST: SERIOUS SYMPTOM CLUSTER TRIAGE
            # Runs BEFORE RAG. If a known serious cluster is detected
            # (diabetes signs, dengue, malaria, thyroid, etc.), generate
            # a targeted response directly — one LLM call, no RAG,
            # no multi-agent loop. RAG would return home-remedy content
            # for these conditions, which is wrong and misleading.
            # ═══════════════════════════════════════════════════════
            if TRIAGE_AVAILABLE:
                cluster = detect_symptom_cluster(query_in_english)
                if cluster:
                    logger.info(
                        f"Fast Triage: cluster '{cluster['id']}' detected — "
                        f"bypassing RAG and multi-agent."
                    )
                    try:
                        cluster_response_text = asyncio.run(
                            generate_cluster_response(
                                cluster=cluster,
                                user_symptoms=query_in_english,
                                bio_context=bio_context_str,
                            )
                        )

                        # Translate and return immediately
                        (
                            translated_response,
                            clean_response,
                            follow_up_questions,
                            _,
                        ) = asyncio.run(
                            postprocess_and_translate_query_response(
                                cluster_response_text,
                                input_language_detected,
                                str(message_id),
                            )
                        )

                        response_data.data["message"] = "Successful retrieval of response"
                        response_data.data["message_id"] = message_id
                        response_data.data["response"] = translated_response
                        response_data.data["follow_up_questions"] = follow_up_questions
                        response_data.data["pipeline"] = "triage_fast_path"
                        response_data.data["triage_cluster"] = cluster["id"]
                        response_data.data["is_emergency_block"] = False
                        response_data.data["has_remedies"] = False
                        response_data.data["agent_verified"] = False
                        response_data.data["bio_state"] = {
                            "circadian_phase": bio_state.circadian_phase.value,
                            "seasonal_influence": bio_state.seasonal_influence.value,
                            "sleep_pressure": bio_state.sleep_pressure_estimate.value,
                            "is_misaligned": bio_state.is_misaligned,
                        }
                        logger.info(
                            f"Fast Triage complete | Cluster: {cluster['id']} | "
                            f"User: {mask_email(email_id)}"
                        )
                        return response_data

                    except Exception as e:
                        logger.error(
                            f"Fast Triage generation failed: {e} — falling back to RAG pipeline",
                            exc_info=True,
                        )
                        # Fall through to normal RAG pipeline on error

            # Inject chronobiology context into the RAG pipeline
            chrono_context = ""
            if bio_state.advisory:
                chrono_context = (
                    f"\n[CHRONOBIOLOGY CONTEXT: {bio_state.advisory}]\n"
                    f"[Circadian Phase: {bio_state.circadian_phase.value}]\n"
                    f"[Sleep Pressure: {bio_state.sleep_pressure_estimate.value}]"
                )

            # Execute RAG pipeline
            try:
                response_pair = asyncio.run(
                    execute_rag_pipeline(
                        query_in_english,
                        input_language_detected,
                        email_id,
                        user_name=user_name,
                        message_id=message_id,
                        chat_history=chat_history,
                        user_profile=user_profile_data,
                    )
                )

                if isinstance(response_pair, tuple) and len(response_pair) == 2:
                    response_map, _ = response_pair
                else:
                    response_map = {
                        "generated_final_response": (
                            "I'm having trouble processing your request right now."
                        )
                    }
            except Exception as e:
                logger.error(f"RAG pipeline failed: {e}", exc_info=True)
                response_map = {
                    "generated_final_response": (
                        "I'm having trouble processing your request right now."
                    )
                }

            llm_response = response_map.get("generated_final_response", "")

            # ═══════════════════════════════════════════════════════
            # STEP C-2: MULTI-AGENT VERIFICATION (Proposer → Critic → Safety)
            # ═══════════════════════════════════════════════════════
            agent_pipeline_used = False
            is_emergency_block = False
            has_remedies = True

            if MULTI_AGENT_AVAILABLE:
                try:
                    # bio_context_str already built above (in STEP C-FAST section)

                    # Pass user profile into the LangGraph loop for deterministic safety.
                    # Pass llm_response as both rag_context (for Proposer fallback) and
                    # initial_draft (so Proposer skips its first LLM call — the RAG
                    # pipeline already generated this draft).
                    # Merge Graph RAG evidence into RAG context for Proposer
                    combined_context = llm_response
                    if kg_context_str:
                        combined_context = f"{llm_response}\n\n{kg_context_str}"

                    verified_response = asyncio.run(
                        run_verification_pipeline(
                            user_symptoms=query_in_english,
                            rag_context=combined_context,
                            bio_context=bio_context_str,
                            medical_profile=user_profile_data or {},
                            initial_draft=llm_response,
                        )
                    )

                    if isinstance(verified_response, dict):
                        llm_response = verified_response.get("response_text", "")
                        is_emergency_block = verified_response.get("is_emergency_block", False)
                        has_remedies = verified_response.get("has_remedies", True)
                        agent_pipeline_used = True
                        logger.info("🤖 Multi-Agent pipeline produced verified response (dict)")
                    elif verified_response:
                        llm_response = verified_response
                        agent_pipeline_used = True
                        logger.info("🤖 Multi-Agent pipeline produced verified response (str)")

                except Exception as e:
                    logger.error(f"Multi-Agent pipeline error: {e}", exc_info=True)
                    logger.info("Falling back to original RAG response")

            # ═══════════════════════════════════════════════════════
            # STEP D-1: TRUST ENGINE VERIFICATION (evidence scoring)
            # ═══════════════════════════════════════════════════════
            trust_data = None
            if TRUST_ENGINE_AVAILABLE and llm_response and has_remedies:
                try:
                    trust_engine = get_trust_engine()
                    # Build user lists for trust engine
                    user_conditions = []
                    user_medications = []
                    user_allergies = []
                    if user_profile_data:
                        user_conditions = user_profile_data.get('medical_conditions', []) or []
                        user_medications = user_profile_data.get('current_medications', []) or []
                        user_allergies = user_profile_data.get('allergies', []) or []

                    trust_result = asyncio.run(
                        trust_engine.verify_response(
                            llm_response=llm_response,
                            query=query_in_english,
                            user_id=email_id,
                            user_conditions=user_conditions,
                            user_medications=user_medications,
                            user_allergies=user_allergies,
                        )
                    )

                    # ALLERGY HARD BLOCK: if any detected herb is an allergen,
                    # suppress the original response and replace with safe alternatives.
                    allergy_blocked_herbs = [
                        h for h in trust_result.contraindicated_herbs
                        if any(a.lower() in h.lower() or h.lower() in a.lower()
                               for a in user_allergies)
                    ]
                    if allergy_blocked_herbs:
                        detected_condition = trust_engine._identify_condition(query_in_english)
                        llm_response = _build_allergy_block_response(
                            allergy_blocked_herbs=allergy_blocked_herbs,
                            condition=detected_condition,
                            user_allergies=user_allergies,
                            user_medications=user_medications,
                            user_conditions=user_conditions,
                            trust_engine=trust_engine,
                        )
                        logger.warning(
                            f"ALLERGY HARD BLOCK: {allergy_blocked_herbs} suppressed "
                            f"for {mask_email(email_id)}"
                        )
                    # Append trust engine's formatted evidence summary to the response
                    elif trust_result.formatted_output:
                        llm_response += trust_result.formatted_output

                    trust_data = {
                        'verified_herbs': trust_result.verified_herbs,
                        'unverified_herbs': trust_result.unverified_herbs,
                        'verified_count': len(trust_result.verified_herbs),
                        'unverified_count': len(trust_result.unverified_herbs),
                        'warnings': trust_result.warnings,
                        'interaction_warnings': trust_result.interaction_warnings,
                        'is_safe': trust_result.is_safe,
                    }
                    logger.info(
                        f"🔬 Trust Engine: {len(trust_result.verified_herbs)} verified, "
                        f"{len(trust_result.unverified_herbs)} unverified"
                    )
                except Exception as e:
                    logger.error(f"Trust Engine error: {e}", exc_info=True)

            # ═══════════════════════════════════════════════════════
            # STEP D-2: TEMPORAL REASONING ENGINE (DB-backed)
            # Runs the full TemporalReasoningEngine for each herb
            # mentioned in the response, using real MedicationHistory
            # rows (with accurate start/stop dates) rather than the
            # Pydantic profile which the multi-agent loop already used.
            # Appends structured temporal warnings to the response.
            # ═══════════════════════════════════════════════════════
            temporal_warnings = []
            if TEMPORAL_ENGINE_AVAILABLE and llm_response and has_remedies and email_id:
                try:
                    temporal_engine = get_temporal_engine()
                    herbs_to_check = _extract_herbs_from_response(llm_response)
                    for herb in herbs_to_check:
                        db_result = asyncio.run(
                            temporal_engine.validate_safety_profile(email_id, herb)
                        )
                        if not db_result.is_safe:
                            temporal_warnings.extend(db_result.warnings)
                            logger.warning(
                                f"⏳ TemporalReasoningEngine flagged '{herb}' "
                                f"for {mask_email(email_id)}: "
                                f"{db_result.warnings}"
                            )
                    if temporal_warnings:
                        warning_block = (
                            "\n\n---\n"
                            "⏳ **Temporal Safety Warnings** *(based on your medication timeline)*\n\n"
                            + "\n".join(f"- {w}" for w in temporal_warnings)
                            + "\n\n*These warnings are generated by ServVia's Temporal "
                            "Pharmacovigilance Engine using your recorded medication history. "
                            "Please consult your doctor before proceeding.*"
                        )
                        llm_response += warning_block
                        logger.info(
                            f"⏳ {len(temporal_warnings)} temporal warning(s) appended to response"
                        )
                except Exception as e:
                    logger.error(f"TemporalReasoningEngine error: {e}", exc_info=True)

            # ═══════════════════════════════════════════════════════
            # STEP E: FINAL OUTPUT
            # ═══════════════════════════════════════════════════════

            # The multi-agent pipeline now handles all safety blocking internally.
            final_response = llm_response
            pipeline_status = "safe_response"

            # Translate final response back to input language
            (
                translated_response,
                clean_response,
                follow_up_questions,
                _,
            ) = asyncio.run(
                postprocess_and_translate_query_response(
                    final_response,
                    input_language_detected,
                    str(message_id),
                )
            )

            # ─── Build response ──────────────────────────────────
            response_data.data["message"] = "Successful retrieval of response"
            response_data.data["message_id"] = message_id
            response_data.data["response"] = translated_response
            response_data.data["source"] = response_map.get("source")
            response_data.data["follow_up_questions"] = follow_up_questions
            response_data.data["pipeline"] = pipeline_status
            response_data.data["is_emergency_block"] = is_emergency_block
            response_data.data["has_remedies"] = has_remedies
            response_data.data["bio_state"] = {
                "circadian_phase": bio_state.circadian_phase.value,
                "seasonal_influence": bio_state.seasonal_influence.value,
                "sleep_pressure": bio_state.sleep_pressure_estimate.value,
                "is_misaligned": bio_state.is_misaligned,
                "weather": bio_state.weather_description,
                "temperature_celsius": bio_state.temperature_celsius,
                "location": f"{bio_state.location_city}, {bio_state.location_country}"
                    if bio_state.location_city else None,
            }

            if trust_data:
                response_data.data["trust_verification"] = trust_data

            if temporal_warnings:
                response_data.data["temporal_warnings"] = temporal_warnings

            if agent_pipeline_used:
                response_data.data["agent_verified"] = True

            logger.info(
                f"✅ Pipeline complete | Status: {pipeline_status} | "
                f"User: {mask_email(email_id)}"
            )

        except Exception as error:
            logger.error(error, exc_info=True)
            response_data.data.update(
                {"message": "Something went wrong", "error": True}
            )
            response_data.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return response_data

    @action(detail=False, methods=["post"])
    def stream(self, request):
        """
        Streaming chat endpoint — runs the full ServVia pipeline then
        streams the final response word-by-word as Server-Sent Events.

        POST /api/chat/stream/
        Body: { "email_id": "...", "query": "..." }

        SSE event types:
            { "type": "metadata", ...pipeline_data }   — sent first (instant)
            { "type": "token", "content": "word " }    — one per word
            { "type": "done" }                         — end of stream
        """
        import json
        import time
        from django.http import StreamingHttpResponse

        email_id = request.data.get("email_id")
        original_query = request.data.get("query")

        # ── Run the full pipeline synchronously first ─────────────────────
        # We reuse get_answer_for_text_query logic by calling it and
        # extracting its response data, then streaming the text portion.
        pipeline_result = {}
        error_response = None

        try:
            # Inline the pipeline — identical to get_answer_for_text_query
            # but we capture the result dict instead of returning a Response.
            authenticated_user = authenticate_user_based_on_email(email_id)
            if not authenticated_user:
                error_response = "Invalid Email ID"
                raise ValueError(error_response)

            if not original_query:
                error_response = "Please submit a query."
                raise ValueError(error_response)

            logger.info(f"🌊 Stream endpoint | User: {mask_email(email_id)}")

            # Step A: Emergency detection
            emergency_type = EmergencySystem.detect_intent(original_query)
            if emergency_type:
                emergency_response = EmergencySystem.get_response(emergency_type)
                pipeline_result = {
                    "response": emergency_response,
                    "pipeline": "emergency_intercept",
                    "emergency_type": emergency_type,
                    "is_emergency_block": False,
                    "has_remedies": False,
                }
            else:
                # Step B: Chronobiology (real IP geolocation + weather)
                client_ip = _get_client_ip(request)
                bio_state = _chrono_engine.infer_state_from_request(client_ip)

                user_data, message_obj = preprocess_user_data(
                    original_query, email_id, authenticated_user
                )
                user_id = user_data.get("user_id")
                user_name = user_data.get("user_name")
                message_id = user_data.get("message_id")
                chat_history = get_user_chat_history(user_id) if user_id else None

                query_in_english, input_language_detected = asyncio.run(
                    detect_language_and_translate_to_english(original_query)
                )

                user_profile_data = None
                try:
                    from user_profile.models import UserProfile
                    profile = UserProfile.objects.get(email=email_id)
                    user_profile_data = {
                        "allergies": profile.get_allergies_list(),
                        "medical_conditions": profile.get_conditions_list(),
                        "current_medications": profile.get_medications_list(),
                        "first_name": profile.first_name,
                    }
                    if profile.first_name:
                        user_name = profile.first_name
                except Exception:
                    pass

                bio_context_str = ""
                if bio_state.advisory:
                    bio_context_str = (
                        f"Circadian Phase: {bio_state.circadian_phase.value}, "
                        f"Season: {bio_state.seasonal_influence.value}, "
                        f"Sleep Pressure: {bio_state.sleep_pressure_estimate.value}"
                    )

                # Step B-KG: Graph RAG adaptive ranking
                kg_context_str = ""
                if GRAPH_RAG_AVAILABLE and email_id:
                    try:
                        allergies_for_kg = (user_profile_data or {}).get("allergies", []) or []
                        meds_for_kg = (user_profile_data or {}).get("current_medications", []) or []
                        _, kg_context_str = _adaptive_ranker.get_top_remedies_for_user(
                            user_email=email_id,
                            condition=query_in_english,
                            exclude_herbs=allergies_for_kg,
                            user_medications=meds_for_kg,
                        )
                    except Exception as e:
                        logger.warning(f"Graph RAG step failed in stream (non-fatal): {e}")

                # Step C-Fast: Triage
                llm_response = None
                is_emergency_block = False
                has_remedies = True
                agent_pipeline_used = False
                triage_fast = False

                if TRIAGE_AVAILABLE:
                    cluster = detect_symptom_cluster(query_in_english)
                    if cluster:
                        try:
                            cluster_response_text = asyncio.run(
                                generate_cluster_response(
                                    cluster=cluster,
                                    user_symptoms=query_in_english,
                                    bio_context=bio_context_str,
                                )
                            )
                            (translated_response, _, follow_up, _) = asyncio.run(
                                postprocess_and_translate_query_response(
                                    cluster_response_text, input_language_detected, str(message_id)
                                )
                            )
                            llm_response = translated_response
                            has_remedies = False
                            triage_fast = True
                        except Exception as e:
                            logger.error(f"Triage failed in stream: {e}", exc_info=True)

                if not triage_fast:
                    # Step C: RAG
                    try:
                        response_pair = asyncio.run(
                            execute_rag_pipeline(
                                query_in_english, input_language_detected, email_id,
                                user_name=user_name, message_id=message_id,
                                chat_history=chat_history, user_profile=user_profile_data,
                            )
                        )
                        response_map = response_pair[0] if isinstance(response_pair, tuple) else {"generated_final_response": ""}
                    except Exception as e:
                        logger.error(f"RAG failed in stream: {e}", exc_info=True)
                        response_map = {"generated_final_response": "I'm having trouble processing your request."}

                    llm_response = response_map.get("generated_final_response", "")

                    # Step C-2: Multi-agent
                    if MULTI_AGENT_AVAILABLE:
                        try:
                            stream_combined_context = llm_response
                            if kg_context_str:
                                stream_combined_context = f"{llm_response}\n\n{kg_context_str}"
                            verified = asyncio.run(
                                run_verification_pipeline(
                                    user_symptoms=query_in_english,
                                    rag_context=stream_combined_context,
                                    bio_context=bio_context_str,
                                    medical_profile=user_profile_data or {},
                                    initial_draft=llm_response,
                                )
                            )
                            if isinstance(verified, dict):
                                llm_response = verified.get("response_text", llm_response)
                                is_emergency_block = verified.get("is_emergency_block", False)
                                has_remedies = verified.get("has_remedies", True)
                            elif verified:
                                llm_response = verified
                            agent_pipeline_used = True
                        except Exception as e:
                            logger.error(f"Multi-agent failed in stream: {e}", exc_info=True)

                    # Step D-1: Trust engine
                    trust_data = None
                    if TRUST_ENGINE_AVAILABLE and llm_response and has_remedies:
                        try:
                            trust_engine = get_trust_engine()
                            user_conditions = (user_profile_data or {}).get("medical_conditions", []) or []
                            user_medications = (user_profile_data or {}).get("current_medications", []) or []
                            user_allergies = (user_profile_data or {}).get("allergies", []) or []
                            trust_result = asyncio.run(
                                trust_engine.verify_response(
                                    llm_response=llm_response, query=query_in_english, user_id=email_id,
                                    user_conditions=user_conditions, user_medications=user_medications,
                                    user_allergies=user_allergies,
                                )
                            )
                            allergy_blocked_herbs = [
                                h for h in trust_result.contraindicated_herbs
                                if any(a.lower() in h.lower() or h.lower() in a.lower()
                                       for a in user_allergies)
                            ]
                            if allergy_blocked_herbs:
                                detected_condition = trust_engine._identify_condition(query_in_english)
                                llm_response = _build_allergy_block_response(
                                    allergy_blocked_herbs=allergy_blocked_herbs,
                                    condition=detected_condition,
                                    user_allergies=user_allergies,
                                    user_medications=user_medications,
                                    user_conditions=user_conditions,
                                    trust_engine=trust_engine,
                                )
                                logger.warning(
                                    f"ALLERGY HARD BLOCK (stream): {allergy_blocked_herbs} "
                                    f"suppressed for {mask_email(email_id)}"
                                )
                            elif trust_result.formatted_output:
                                llm_response += trust_result.formatted_output
                            trust_data = {
                                "verified_count": len(trust_result.verified_herbs),
                                "unverified_count": len(trust_result.unverified_herbs),
                                "interaction_warnings": trust_result.interaction_warnings,
                                "is_safe": trust_result.is_safe,
                            }
                        except Exception as e:
                            logger.error(f"Trust engine failed in stream: {e}", exc_info=True)

                    # Step D-2: Temporal engine
                    if TEMPORAL_ENGINE_AVAILABLE and llm_response and has_remedies:
                        try:
                            temporal_engine = get_temporal_engine()
                            herbs = _extract_herbs_from_response(llm_response)
                            temporal_warnings = []
                            for herb in herbs:
                                db_result = asyncio.run(temporal_engine.validate_safety_profile(email_id, herb))
                                if not db_result.is_safe:
                                    temporal_warnings.extend(db_result.warnings)
                            if temporal_warnings:
                                llm_response += (
                                    "\n\n---\n⏳ **Temporal Safety Warnings**\n\n"
                                    + "\n".join(f"- {w}" for w in temporal_warnings)
                                )
                        except Exception as e:
                            logger.error(f"Temporal engine failed in stream: {e}", exc_info=True)

                    # Step E: Translate final response
                    (translated_response, _, follow_up, _) = asyncio.run(
                        postprocess_and_translate_query_response(
                            llm_response, input_language_detected, str(message_id)
                        )
                    )
                    llm_response = translated_response

                    pipeline_result = {
                        "response": llm_response,
                        "pipeline": "safe_response",
                        "is_emergency_block": is_emergency_block,
                        "has_remedies": has_remedies,
                        "bio_state": {
                            "circadian_phase": bio_state.circadian_phase.value,
                            "seasonal_influence": bio_state.seasonal_influence.value,
                            "sleep_pressure": bio_state.sleep_pressure_estimate.value,
                            "is_misaligned": bio_state.is_misaligned,
                        },
                        "agent_verified": agent_pipeline_used,
                        **({"trust_verification": trust_data} if trust_data else {}),
                    }
                else:
                    # Triage fast-path result
                    pipeline_result = {
                        "response": llm_response,
                        "pipeline": "triage_fast_path",
                        "is_emergency_block": False,
                        "has_remedies": False,
                        "bio_state": {
                            "circadian_phase": bio_state.circadian_phase.value,
                            "seasonal_influence": bio_state.seasonal_influence.value,
                            "sleep_pressure": bio_state.sleep_pressure_estimate.value,
                            "is_misaligned": bio_state.is_misaligned,
                        },
                    }

        except ValueError:
            pipeline_result = {"response": error_response or "An error occurred.", "pipeline": "error"}
        except Exception as e:
            logger.error(f"Stream endpoint error: {e}", exc_info=True)
            pipeline_result = {"response": "Something went wrong. Please try again.", "pipeline": "error"}

        # ── SSE generator ─────────────────────────────────────────────────
        response_text = pipeline_result.pop("response", "")

        def sse_stream():
            # 1. Metadata chunk (pipeline badges, bio state, trust, etc.)
            meta_event = json.dumps({"type": "metadata", **pipeline_result})
            yield f"data: {meta_event}\n\n"

            # 2. Stream response word-by-word for the "typing" effect
            # Split on spaces but preserve them in the output
            words = response_text.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                token_event = json.dumps({"type": "token", "content": token})
                yield f"data: {token_event}\n\n"
                time.sleep(0.018)  # ~55 words/sec — feels natural, not robotic

            # 3. Done
            yield "data: {\"type\": \"done\"}\n\n"

        resp = StreamingHttpResponse(sse_stream(), content_type="text/event-stream")
        resp["Cache-Control"] = "no-cache"
        resp["X-Accel-Buffering"] = "no"
        return resp

    @action(detail=False, methods=["post"])
    def synthesise_audio(self, request):
        """Proxy to legacy TTS endpoint."""
        from legacy_agriculture.api.views import ChatAPIViewSet
        legacy = ChatAPIViewSet()
        return legacy.synthesise_audio(request)

    @action(detail=False, methods=["post"])
    def transcribe_audio(self, request):
        """Proxy to legacy ASR endpoint."""
        from legacy_agriculture.api.views import ChatAPIViewSet
        legacy = ChatAPIViewSet()
        return legacy.transcribe_audio(request)

    @action(detail=False, methods=["post"])
    def get_answer_by_voice_query(self, request):
        """Proxy to legacy voice endpoint."""
        from legacy_agriculture.api.views import ChatAPIViewSet
        legacy = ChatAPIViewSet()
        return legacy.get_answer_by_voice_query(request)
