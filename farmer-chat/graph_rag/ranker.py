"""
ServVia Graph RAG — Outcome Adaptive Ranker
============================================

Implements Objective 4: Outcome-Adaptive Remedy Ranking via Lightweight Graph RAG.

The OutcomeAdaptiveRanker:
    1. Reads UserRemedyOutcome records for the requesting user.
    2. Computes per-herb personalization weight multipliers.
    3. Applies weights to IntegrativeKnowledgeGraph confidence scores.
    4. Re-sorts candidates so personalized herbs rank higher/lower.
    5. Formats top results as structured context for the Proposer agent.

This is "episodic" personalization — it refines future recommendations
without engaging in invasive longitudinal behavioral surveillance.
No raw behavioral data is transmitted to LLMs; only adjusted ranking order.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger("ServVia.OutcomeAdaptiveRanker")


class OutcomeAdaptiveRanker:
    """
    Reranks KG evidence candidates using episodic user feedback.

    Works on top of IntegrativeKnowledgeGraph.get_evidence_for_condition()
    output. Applies personalization weights without modifying the static
    knowledge graph — all adjustments are per-user, per-request.
    """

    def get_personalization_weights(self, user_email: str) -> Dict[str, float]:
        """
        Compute a weight multiplier for each herb the user has given feedback on.

        Multiple outcomes for the same herb are averaged.

        Args:
            user_email: The user's email identifier.

        Returns:
            {herb_name_lowercase: average_weight_multiplier}
            Empty dict if no feedback exists or DB unavailable.
        """
        try:
            from graph_rag.models import UserRemedyOutcome
            outcomes = UserRemedyOutcome.objects.filter(user_email=user_email)
        except Exception as exc:
            logger.warning(f"Could not load outcomes for {user_email}: {exc}")
            return {}

        weights: Dict[str, List[float]] = {}
        for outcome in outcomes:
            herb = outcome.remedy_name.lower().strip()
            weights.setdefault(herb, []).append(outcome.personalization_weight)

        averaged = {herb: sum(ws) / len(ws) for herb, ws in weights.items()}
        if averaged:
            logger.info(
                f"🎯 Personalization weights loaded | User: {user_email} | "
                f"Herbs: {list(averaged.keys())}"
            )
        return averaged

    def rerank(
        self,
        candidates: List[Dict],
        user_email: str,
    ) -> List[Dict]:
        """
        Apply personalization weights to KG candidates and re-sort.

        Args:
            candidates: Output of IntegrativeKnowledgeGraph.get_evidence_for_condition().
                        Each dict has 'herb' (HerbNode), 'evidence' (EvidenceEdge), etc.
            user_email: User identifier for weight lookup.

        Returns:
            Candidates re-sorted by personalized_score (descending),
            with 'personalized_score' and 'personalization_applied' fields added.
        """
        weights = self.get_personalization_weights(user_email)

        applied = 0
        for c in candidates:
            herb_name = c["herb"].common_name.lower()
            weight = weights.get(herb_name, 1.0)
            base_score = c["evidence"].confidence_score
            c["personalized_score"] = round(base_score * weight, 2)
            c["personalization_applied"] = herb_name in weights
            if herb_name in weights:
                applied += 1

        candidates.sort(key=lambda x: x.get("personalized_score", 0), reverse=True)

        if applied:
            logger.info(
                f"🎯 Adaptive reranking complete | {applied}/{len(candidates)} herbs adjusted"
            )

        return candidates

    def format_kg_context(
        self,
        ranked_candidates: List[Dict],
        max_results: int = 3,
    ) -> str:
        """
        Format top-N ranked KG results as a context string for the Proposer agent.

        This string is injected alongside the RAG response to give the
        multi-agent pipeline evidence-graded, personalized remedy options.

        Args:
            ranked_candidates: Output of rerank().
            max_results: Maximum number of remedies to include.

        Returns:
            Formatted markdown string, or empty string if no candidates.
        """
        if not ranked_candidates:
            return ""

        lines = ["📊 **Evidence-Graded Remedies (Graph RAG)**\n"]
        for c in ranked_candidates[:max_results]:
            herb = c["herb"]
            edge = c["evidence"]
            badge = c["badge"]
            personalized = c.get("personalization_applied", False)
            personal_tag = " *(personalized for you)*" if personalized else ""
            score = c.get("personalized_score", edge.confidence_score)

            mechanism = edge.mechanism_description
            if mechanism and len(mechanism) > 120:
                mechanism = mechanism[:117] + "..."

            lines.append(
                f"**{herb.common_name}** {badge['icon']} {badge['label']}{personal_tag}\n"
                f"- Mechanism: {mechanism}\n"
                f"- Dose: {edge.recommended_dose or 'standard'} | "
                f"Onset: {edge.onset_time or 'varies'}\n"
                f"- Evidence score: {score}/10\n"
            )

        return "\n".join(lines)

    def get_top_remedies_for_user(
        self,
        user_email: str,
        condition: str,
        exclude_herbs: Optional[List[str]] = None,
        user_medications: Optional[List[str]] = None,
        min_tier: int = 4,
    ) -> tuple[List[Dict], str]:
        """
        Convenience method: query the KG, apply adaptive reranking, and format.

        Args:
            user_email: User identifier.
            condition: Condition string (used to query KG).
            exclude_herbs: Allergen exclusion list.
            user_medications: Current medications for interaction check.
            min_tier: Minimum evidence tier (1=clinical, 4=anecdotal).

        Returns:
            (ranked_candidates, formatted_context_string)
        """
        try:
            from servvia2.knowledge_graph.integrative_kg import IntegrativeKnowledgeGraph
            kg = IntegrativeKnowledgeGraph()
            candidates = kg.get_evidence_for_condition(
                condition=condition,
                exclude_herbs=exclude_herbs or [],
                user_medications=user_medications or [],
                min_tier=min_tier,
            )
        except Exception as exc:
            logger.warning(f"KG query failed for '{condition}': {exc}")
            return [], ""

        if not candidates:
            return [], ""

        ranked = self.rerank(candidates, user_email)
        context = self.format_kg_context(ranked)
        return ranked, context
