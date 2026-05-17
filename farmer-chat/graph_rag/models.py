"""
ServVia Graph RAG — UserRemedyOutcome Model
============================================

Stores episodic user feedback on remedy efficacy.
Used by OutcomeAdaptiveRanker to dynamically reweight
Graph RAG retrieval rankings (Objective 4).

Edge weight adjustment formula:
    personalized_score = base_confidence_score × avg(personalization_weight)

Weight derivation from efficacy_score (1–5):
    1/5 → 0.50  (strong demotion — remedy did not work)
    2/5 → 0.75  (mild demotion)
    3/5 → 1.00  (neutral — no change to ranking)
    4/5 → 1.25  (mild promotion)
    5/5 → 1.50  (strong promotion — remedy worked very well)
"""

from django.db import models


class UserRemedyOutcome(models.Model):
    """
    Single episodic outcome record for a remedy tried by a user.
    """

    EFFICACY_CHOICES = [
        (1, "1 — No effect"),
        (2, "2 — Minimal improvement"),
        (3, "3 — Moderate improvement"),
        (4, "4 — Good improvement"),
        (5, "5 — Excellent — fully resolved"),
    ]

    user_email = models.EmailField(db_index=True)
    remedy_name = models.CharField(
        max_length=200,
        db_index=True,
        help_text="Herb or remedy name, stored lowercase.",
    )
    condition_treated = models.CharField(
        max_length=300,
        help_text="The symptom or condition the remedy was used for.",
    )
    efficacy_score = models.IntegerField(
        choices=EFFICACY_CHOICES,
        help_text="1–5 scale of how well the remedy worked.",
    )
    improvement_noted = models.BooleanField(
        default=False,
        help_text="Quick boolean: did the user notice improvement?",
    )
    side_effects_reported = models.TextField(
        blank=True,
        help_text="Free-text description of any side effects experienced.",
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional user notes about the remedy experience.",
    )
    tried_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Timestamp when feedback was submitted.",
    )

    class Meta:
        db_table = "user_remedy_outcomes"
        ordering = ["-tried_at"]
        indexes = [
            models.Index(fields=["user_email", "remedy_name"]),
        ]

    def __str__(self) -> str:
        return f"{self.user_email} | {self.remedy_name} | score={self.efficacy_score}"

    @property
    def personalization_weight(self) -> float:
        """
        Convert 1–5 efficacy score to a retrieval weight multiplier.

        Linear mapping:
            score 1 → 0.50  (demote heavily)
            score 3 → 1.00  (neutral)
            score 5 → 1.50  (promote heavily)
        """
        return 0.5 + (self.efficacy_score - 1) * 0.25
