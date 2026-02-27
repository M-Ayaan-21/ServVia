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

# Trust Engine for scientific validation
try:
    from servvia2.trust_engine.engine import get_trust_engine
    TRUST_ENGINE_AVAILABLE = True
except ImportError:
    TRUST_ENGINE_AVAILABLE = False

# Multi-Agent LangGraph verification
try:
    from agents.graph import run_verification_pipeline
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-Agent graph not available: {e}")
    MULTI_AGENT_AVAILABLE = False

logger = logging.getLogger("ServVia.PipelineView")

# ─── Singleton instances (initialized once at import time) ──────────────
_chrono_engine = ChronobiologyEngine()
_safety_validator = TemporalSafetyValidator()


# =============================================================================
# HELPER: BUILD MEDICAL PROFILE FROM USER PROFILE DATA
# =============================================================================

def _build_medical_profile(email_id: str, profile_data: dict = None) -> UserMedicalProfile:
    """
    Convert the Django UserProfile data into a typed Pydantic
    UserMedicalProfile for the TemporalSafetyValidator.

    If no profile data is available, returns a minimal empty profile.

    Args:
        email_id: User's email for identification.
        profile_data: Dict from UserProfile model (allergies, medications, etc.)

    Returns:
        UserMedicalProfile: Typed, validated profile.
    """
    if not profile_data:
        return UserMedicalProfile(
            user_id=email_id,
            allergies=[],
            current_medications=[],
            symptom_onset_hours=0,
        )

    # Convert medication strings to MedicationRecord objects
    med_records = []
    raw_meds = profile_data.get("current_medications", []) or []
    for med_name in raw_meds:
        if isinstance(med_name, str) and med_name.strip():
            med_records.append(
                MedicationRecord(
                    drug_name=med_name.strip(),
                    start_date=datetime.now(timezone.utc) - timedelta(days=90),
                    end_date=None,  # Assume active
                )
            )

    return UserMedicalProfile(
        user_id=email_id,
        allergies=profile_data.get("allergies", []) or [],
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
        f"⚠️ **Safety Alert — Recommendation Blocked**\n\n"
        f"I was about to recommend **{herb_name}**, but our safety system "
        f"has flagged a potential issue:\n\n"
        f"---\n\n"
        f"**{validation_result.reason}**\n\n"
        f"---\n\n"
    )

    if validation_result.washout_days_remaining:
        block_response += (
            f"⏳ **Washout Period:** {validation_result.washout_days_remaining} "
            f"day(s) remaining before this remedy is considered safe.\n\n"
        )

    if validation_result.contraindications:
        block_response += "**Specific Contraindications:**\n"
        for ci in validation_result.contraindications:
            block_response += f"  • {ci}\n"
        block_response += "\n"

    block_response += (
        "💡 **What you can do:**\n"
        "  • Consult your doctor before trying herbal remedies with your current medications.\n"
        "  • Ask me about alternative remedies that are safe with your profile.\n"
        "  • Update your medication list in your profile if it's changed.\n\n"
        "🔒 *This safety check is powered by ServVia's Neurosymbolic "
        "Pharmacovigilance Engine — a deterministic, evidence-based system.*"
    )

    return block_response


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
            # ═══════════════════════════════════════════════════════
            bio_state = _chrono_engine.infer_state(
                local_time=datetime.now(timezone(timedelta(hours=5, minutes=30))),  # IST
            )
            logger.info(
                f"🕐 Bio State: phase={bio_state.circadian_phase.value}, "
                f"season={bio_state.seasonal_influence.value}, "
                f"misaligned={bio_state.is_misaligned}"
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

            # Inject chronobiology context into the RAG pipeline
            # The bio_state advisory will be appended to the query context
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
            # STEP C-2: MULTI-AGENT VERIFICATION (Proposer → Critic)
            # ═══════════════════════════════════════════════════════
            agent_pipeline_used = False
            if MULTI_AGENT_AVAILABLE:
                try:
                    bio_context_str = ""
                    if bio_state.advisory:
                        bio_context_str = (
                            f"Circadian Phase: {bio_state.circadian_phase.value}, "
                            f"Season: {bio_state.seasonal_influence.value}, "
                            f"Sleep Pressure: {bio_state.sleep_pressure_estimate.value}"
                        )

                    # The RAG response becomes context for the Proposer
                    verified_response = asyncio.run(
                        run_verification_pipeline(
                            user_symptoms=query_in_english,
                            rag_context=llm_response,  # RAG output as knowledge
                            bio_context=bio_context_str,
                        )
                    )

                    if verified_response:
                        llm_response = verified_response
                        agent_pipeline_used = True
                        logger.info("🤖 Multi-Agent pipeline produced verified response")

                except Exception as e:
                    logger.error(f"Multi-Agent pipeline error: {e}", exc_info=True)
                    logger.info("Falling back to original RAG response")

            # ═══════════════════════════════════════════════════════
            # STEP D-1: TRUST ENGINE VERIFICATION (evidence scoring)
            # ═══════════════════════════════════════════════════════
            trust_data = None
            if TRUST_ENGINE_AVAILABLE and llm_response:
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

                    # Append trust engine's formatted evidence summary to the response
                    if trust_result.formatted_output:
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
            # STEP D-2: SAFETY VALIDATION (deterministic, no LLM)
            # ═══════════════════════════════════════════════════════
            safety_blocked = False
            safety_result = None
            blocked_herb = None

            # Build typed medical profile
            medical_profile = _build_medical_profile(email_id, user_profile_data)

            # Extract herbs from the LLM response
            proposed_herbs = _extract_herbs_from_response(llm_response)
            logger.info(f"🔍 Herbs in response: {proposed_herbs}")

            # Validate each proposed herb against the patient's profile
            for herb_name in proposed_herbs:
                proposal = RemedyProposal(
                    herb_or_remedy_name=herb_name,
                    intended_effect="LLM-recommended remedy",
                )
                result = _safety_validator.validate_remedy(
                    medical_profile, proposal
                )

                if not result.is_safe:
                    safety_blocked = True
                    safety_result = result
                    blocked_herb = herb_name
                    logger.warning(
                        f"⛔ SAFETY BLOCK: {herb_name} blocked for "
                        f"{mask_email(email_id)} — {result.reason[:100]}"
                    )
                    break  # Block on first unsafe herb

            # ═══════════════════════════════════════════════════════
            # STEP E: FINAL OUTPUT
            # ═══════════════════════════════════════════════════════
            if safety_blocked and safety_result:
                # Replace the LLM response with a safety block message
                final_response = _format_safety_block(
                    llm_response, safety_result, blocked_herb
                )
                pipeline_status = "safety_blocked"
            else:
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
            response_data.data["bio_state"] = {
                "circadian_phase": bio_state.circadian_phase.value,
                "seasonal_influence": bio_state.seasonal_influence.value,
                "sleep_pressure": bio_state.sleep_pressure_estimate.value,
                "is_misaligned": bio_state.is_misaligned,
            }

            if safety_blocked:
                response_data.data["safety"] = {
                    "is_safe": False,
                    "blocked_herb": blocked_herb,
                    "reason": safety_result.reason,
                    "contraindications": safety_result.contraindications,
                    "washout_days_remaining": safety_result.washout_days_remaining,
                }

            if trust_data:
                response_data.data["trust_verification"] = trust_data

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
