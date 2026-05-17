"""
ServVia Graph RAG — Feedback API
==================================

REST endpoints for capturing episodic remedy outcome feedback.
This data drives the OutcomeAdaptiveRanker, which re-weights
future Graph RAG retrieval results for that user.

Endpoints:
    POST /api/feedback/feedback/log_outcome/
    GET  /api/feedback/feedback/my_outcomes/
"""

import logging

from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from graph_rag.models import UserRemedyOutcome
from graph_rag.ranker import OutcomeAdaptiveRanker

logger = logging.getLogger("ServVia.FeedbackViewSet")


class FeedbackViewSet(GenericViewSet):
    """
    Handles remedy outcome feedback for adaptive Graph RAG re-ranking.
    Unauthenticated — email-based identification (matching platform pattern).
    """

    authentication_classes = []
    permission_classes = []

    @action(detail=False, methods=["post"])
    def log_outcome(self, request):
        """
        Log a remedy outcome feedback record.

        POST /api/feedback/feedback/log_outcome/
        Body:
            {
                "email": "user@example.com",
                "remedy_name": "ginger",
                "condition_treated": "nausea",
                "efficacy_score": 4,          // integer 1–5
                "improvement_noted": true,
                "side_effects_reported": "",
                "notes": "Helped a lot with the nausea"
            }

        Returns:
            {
                "message": "Outcome recorded...",
                "outcome_id": 12,
                "personalization_weight": 1.25
            }
        """
        email = (request.data.get("email") or "").strip().lower()
        remedy_name = (request.data.get("remedy_name") or "").strip().lower()
        condition_treated = (request.data.get("condition_treated") or "").strip()
        efficacy_score = request.data.get("efficacy_score")
        improvement_noted = request.data.get("improvement_noted", False)
        side_effects = (request.data.get("side_effects_reported") or "").strip()
        notes = (request.data.get("notes") or "").strip()

        # Validation
        if not email or not remedy_name or efficacy_score is None:
            return Response(
                {"error": "email, remedy_name, and efficacy_score are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            efficacy_score = int(efficacy_score)
            if not 1 <= efficacy_score <= 5:
                raise ValueError
        except (ValueError, TypeError):
            return Response(
                {"error": "efficacy_score must be an integer between 1 and 5."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        outcome = UserRemedyOutcome.objects.create(
            user_email=email,
            remedy_name=remedy_name,
            condition_treated=condition_treated,
            efficacy_score=efficacy_score,
            improvement_noted=bool(improvement_noted),
            side_effects_reported=side_effects,
            notes=notes,
        )

        logger.info(
            f"📝 Outcome logged | User: {email} | Remedy: {remedy_name} | "
            f"Score: {efficacy_score}/5 | Improvement: {improvement_noted}"
        )

        return Response(
            {
                "message": "Outcome recorded. Thank you — this helps personalise your future recommendations.",
                "outcome_id": outcome.pk,
                "personalization_weight": outcome.personalization_weight,
            },
            status=status.HTTP_201_CREATED,
        )

    @action(detail=False, methods=["get"])
    def my_outcomes(self, request):
        """
        Retrieve a user's remedy outcome history and current personalization weights.

        GET /api/feedback/feedback/my_outcomes/?email=user@example.com

        Returns:
            {
                "email": "...",
                "total_outcomes": 3,
                "personalization_weights": {"ginger": 1.25, "turmeric": 0.75},
                "outcomes": [...]
            }
        """
        email = (request.query_params.get("email") or "").strip().lower()
        if not email:
            return Response(
                {"error": "email query parameter is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        outcomes = UserRemedyOutcome.objects.filter(user_email=email).order_by("-tried_at")
        ranker = OutcomeAdaptiveRanker()
        weights = ranker.get_personalization_weights(email)

        data = [
            {
                "id": o.pk,
                "remedy_name": o.remedy_name,
                "condition_treated": o.condition_treated,
                "efficacy_score": o.efficacy_score,
                "improvement_noted": o.improvement_noted,
                "side_effects_reported": o.side_effects_reported,
                "notes": o.notes,
                "tried_at": o.tried_at.isoformat(),
                "personalization_weight": o.personalization_weight,
            }
            for o in outcomes
        ]

        return Response(
            {
                "email": email,
                "total_outcomes": len(data),
                "personalization_weights": weights,
                "outcomes": data,
            }
        )
