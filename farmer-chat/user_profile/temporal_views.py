"""
ServVia Temporal API Endpoints
==============================

REST endpoints for logging and querying temporal medical data:
    - Medication history (start/stop dates, dosage, status)
    - Symptom onset tracking
    - Real-time temporal safety check

All endpoints identify the user by email (matching legacy auth pattern).

Author: ServVia Engineering
Version: 1.0.0
"""

import logging
from datetime import datetime

from django.utils import timezone
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet

from .models import AllergyHistory, MedicationHistory, SymptomOnset, UserProfile

logger = logging.getLogger("ServVia.TemporalViews")


def _get_user_or_error(email):
    """Return (UserProfile, None) or (None, Response) if not found."""
    if not email:
        return None, Response(
            {"error": True, "message": "email is required"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    try:
        return UserProfile.objects.get(email=email), None
    except UserProfile.DoesNotExist:
        return None, Response(
            {"error": True, "message": f"No profile found for '{email}'"},
            status=status.HTTP_404_NOT_FOUND,
        )


class TemporalViewSet(GenericViewSet):
    """
    Temporal pharmacovigilance endpoints.

    Mounted at: /api/temporal/
    """

    authentication_classes = []
    permission_classes = []

    # =========================================================================
    # POST /api/temporal/log_medication/
    # =========================================================================

    @action(detail=False, methods=["post"])
    def log_medication(self, request):
        """
        Log a medication start or update an existing record to stopped.

        Body fields:
            email             (str, required)  — user identifier
            medication_name   (str, required)
            dosage            (str, required)
            frequency         (str, required)  — e.g. "twice daily"
            start_date        (str, required)  — ISO 8601, e.g. "2026-01-15"
            stop_date         (str, optional)  — ISO 8601; omit if still active
            status            (str, optional)  — active|discontinued|paused|completed
            generic_name      (str, optional)
            route             (str, optional)  — oral|topical|injection (default: oral)
            prescribed_by     (str, optional)
            reason_for_taking (str, optional)
            reason_for_discontinuation (str, optional)
            side_effects_experienced   (str, optional)

        Returns 201 on creation, 200 if an existing active record is updated.
        """
        email = request.data.get("email")
        user, err = _get_user_or_error(email)
        if err:
            return err

        medication_name = request.data.get("medication_name", "").strip()
        dosage = request.data.get("dosage", "").strip()
        frequency = request.data.get("frequency", "").strip()
        start_date_raw = request.data.get("start_date")

        if not medication_name:
            return Response(
                {"error": True, "message": "medication_name is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not start_date_raw:
            return Response(
                {"error": True, "message": "start_date is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Parse dates
        try:
            start_date = datetime.fromisoformat(start_date_raw)
            if start_date.tzinfo is None:
                start_date = timezone.make_aware(start_date)
        except ValueError:
            return Response(
                {"error": True, "message": f"Invalid start_date format: '{start_date_raw}'. Use ISO 8601."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        stop_date = None
        stop_date_raw = request.data.get("stop_date")
        if stop_date_raw:
            try:
                stop_date = datetime.fromisoformat(stop_date_raw)
                if stop_date.tzinfo is None:
                    stop_date = timezone.make_aware(stop_date)
            except ValueError:
                return Response(
                    {"error": True, "message": f"Invalid stop_date format: '{stop_date_raw}'. Use ISO 8601."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        med_status = request.data.get("status", "active")
        if med_status not in ("active", "discontinued", "paused", "completed"):
            med_status = "active"

        # If stop_date given and status still 'active', set to discontinued
        if stop_date and med_status == "active":
            med_status = "discontinued"

        # Check for an existing active record for the same medication
        existing = MedicationHistory.objects.filter(
            user=user,
            medication_name__iexact=medication_name,
            status__in=("active", "paused"),
        ).first()

        if existing:
            # Update existing record (e.g., user stopped the medication)
            existing.stop_date = stop_date
            existing.status = med_status
            if request.data.get("reason_for_discontinuation"):
                existing.reason_for_discontinuation = request.data["reason_for_discontinuation"]
            if request.data.get("side_effects_experienced"):
                existing.side_effects_experienced = request.data["side_effects_experienced"]
            existing.save()
            logger.info(f"Updated medication record for {email}: {medication_name} → {med_status}")
            return Response(
                {
                    "success": True,
                    "action": "updated",
                    "id": existing.pk,
                    "medication_name": existing.medication_name,
                    "status": existing.status,
                    "stop_date": existing.stop_date.isoformat() if existing.stop_date else None,
                },
                status=status.HTTP_200_OK,
            )

        # Create new record
        med = MedicationHistory.objects.create(
            user=user,
            medication_name=medication_name,
            generic_name=request.data.get("generic_name", ""),
            dosage=dosage,
            frequency=frequency,
            route=request.data.get("route", "oral"),
            start_date=start_date,
            stop_date=stop_date,
            status=med_status,
            prescribed_by=request.data.get("prescribed_by", ""),
            reason_for_taking=request.data.get("reason_for_taking", ""),
            reason_for_discontinuation=request.data.get("reason_for_discontinuation", ""),
            side_effects_experienced=request.data.get("side_effects_experienced", ""),
        )
        logger.info(f"Created medication record for {email}: {medication_name} ({med_status})")
        return Response(
            {
                "success": True,
                "action": "created",
                "id": med.pk,
                "medication_name": med.medication_name,
                "status": med.status,
                "start_date": med.start_date.isoformat(),
                "stop_date": med.stop_date.isoformat() if med.stop_date else None,
            },
            status=status.HTTP_201_CREATED,
        )

    # =========================================================================
    # GET /api/temporal/medication_history/
    # =========================================================================

    @action(detail=False, methods=["get"])
    def medication_history(self, request):
        """
        Retrieve a user's full medication timeline.

        Query params:
            email  (str, required)
            status (str, optional) — filter by active|discontinued|paused|completed

        Returns a list of medication records ordered by start_date descending.
        """
        email = request.query_params.get("email")
        user, err = _get_user_or_error(email)
        if err:
            return err

        qs = MedicationHistory.objects.filter(user=user)
        status_filter = request.query_params.get("status")
        if status_filter:
            qs = qs.filter(status=status_filter)

        records = []
        for med in qs.order_by("-start_date"):
            records.append(
                {
                    "id": med.pk,
                    "medication_name": med.medication_name,
                    "generic_name": med.generic_name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "route": med.route,
                    "start_date": med.start_date.isoformat(),
                    "stop_date": med.stop_date.isoformat() if med.stop_date else None,
                    "status": med.status,
                    "days_since_stopped": med.days_since_stopped(),
                    "is_recently_started": med.is_recently_started(),
                    "prescribed_by": med.prescribed_by,
                    "reason_for_taking": med.reason_for_taking,
                    "reason_for_discontinuation": med.reason_for_discontinuation,
                    "side_effects_experienced": med.side_effects_experienced,
                }
            )

        return Response({"email": email, "count": len(records), "medications": records})

    # =========================================================================
    # POST /api/temporal/log_symptom/
    # =========================================================================

    @action(detail=False, methods=["post"])
    def log_symptom(self, request):
        """
        Log a symptom onset event for acuity tracking.

        Body fields:
            email                (str, required)
            symptom_description  (str, required)
            body_system          (str, required) — e.g. "respiratory", "cardiovascular"
            onset_date           (str, required) — ISO 8601
            severity_at_onset    (int, required) — 1-10 scale
            pattern              (str, required) — constant|intermittent|worsening|improving|fluctuating
            current_severity     (int, optional) — 1-10
            duration_days        (int, optional) — override if known
            is_chronic           (bool, optional)
            associated_conditions (str, optional) — comma-separated or JSON list
            medications_tried    (str, optional) — comma-separated or JSON list

        Returns 201 with the created record and its acuity classification.
        """
        email = request.data.get("email")
        user, err = _get_user_or_error(email)
        if err:
            return err

        symptom_description = request.data.get("symptom_description", "").strip()
        body_system = request.data.get("body_system", "").strip()
        onset_date_raw = request.data.get("onset_date")
        severity_raw = request.data.get("severity_at_onset")
        pattern = request.data.get("pattern", "constant")

        if not symptom_description:
            return Response(
                {"error": True, "message": "symptom_description is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if not onset_date_raw:
            return Response(
                {"error": True, "message": "onset_date is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        if severity_raw is None:
            return Response(
                {"error": True, "message": "severity_at_onset (1-10) is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            onset_date = datetime.fromisoformat(onset_date_raw)
            if onset_date.tzinfo is None:
                onset_date = timezone.make_aware(onset_date)
        except ValueError:
            return Response(
                {"error": True, "message": f"Invalid onset_date format: '{onset_date_raw}'. Use ISO 8601."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            severity_at_onset = int(severity_raw)
            if not 1 <= severity_at_onset <= 10:
                raise ValueError
        except (ValueError, TypeError):
            return Response(
                {"error": True, "message": "severity_at_onset must be an integer between 1 and 10"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        current_severity_raw = request.data.get("current_severity")
        current_severity = None
        if current_severity_raw is not None:
            try:
                current_severity = int(current_severity_raw)
            except (ValueError, TypeError):
                current_severity = None

        if pattern not in ("constant", "intermittent", "worsening", "improving", "fluctuating"):
            pattern = "constant"

        symptom = SymptomOnset.objects.create(
            user=user,
            symptom_description=symptom_description,
            body_system=body_system,
            onset_date=onset_date,
            severity_at_onset=severity_at_onset,
            current_severity=current_severity,
            duration_days=request.data.get("duration_days"),
            is_chronic=bool(request.data.get("is_chronic", False)),
            pattern=pattern,
            associated_conditions=str(request.data.get("associated_conditions", "")),
            medications_tried=str(request.data.get("medications_tried", "")),
        )

        logger.info(f"Logged symptom for {email}: {symptom_description} — acuity: {symptom.acuity_classification()}")
        return Response(
            {
                "success": True,
                "id": symptom.pk,
                "symptom_description": symptom.symptom_description,
                "onset_date": symptom.onset_date.isoformat(),
                "days_since_onset": symptom.days_since_onset(),
                "acuity": symptom.acuity_classification(),
                "is_chronic": symptom.is_chronic,
            },
            status=status.HTTP_201_CREATED,
        )

    # =========================================================================
    # GET /api/temporal/safety_check/
    # =========================================================================

    @action(detail=False, methods=["get"])
    def safety_check(self, request):
        """
        Real-time temporal safety check for a proposed herb.

        Calls both the deterministic TemporalSafetyValidator (Pydantic,
        fast, in-memory) and the full TemporalReasoningEngine (DB-backed,
        includes stabilization windows and allergy cross-reactivity).

        Query params:
            email  (str, required) — user identifier
            herb   (str, required) — e.g. "turmeric", "ginger"

        Returns a merged safety verdict with detailed reasoning.
        """
        email = request.query_params.get("email")
        herb_name = (request.query_params.get("herb") or "").strip()

        user, err = _get_user_or_error(email)
        if err:
            return err

        if not herb_name:
            return Response(
                {"error": True, "message": "herb query param is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ── Build Pydantic profile from real MedicationHistory records ──
        from datetime import timezone as tz
        from core.models import MedicationRecord, RemedyProposal, UserMedicalProfile
        from neurosymbolic.temporal_validator import TemporalSafetyValidator

        med_records = []
        for med in MedicationHistory.objects.filter(user=user):
            start = med.start_date
            if start.tzinfo is None:
                start = start.replace(tzinfo=tz.utc)
            stop = None
            if med.stop_date:
                stop = med.stop_date
                if stop.tzinfo is None:
                    stop = stop.replace(tzinfo=tz.utc)
            med_records.append(
                MedicationRecord(
                    drug_name=med.medication_name,
                    start_date=start,
                    end_date=stop,
                )
            )

        profile = UserMedicalProfile(
            user_id=email,
            allergies=user.get_allergies_list(),
            current_medications=med_records,
            symptom_onset_hours=0,
        )

        proposal = RemedyProposal(
            herb_or_remedy_name=herb_name,
            intended_effect="user-requested safety check",
        )

        validator = TemporalSafetyValidator()
        fast_result = validator.validate_remedy(profile, proposal)

        # ── DB-backed TemporalReasoningEngine check ──
        import asyncio
        from servvia2.temporal_reasoning.engine import get_temporal_engine

        engine = get_temporal_engine()
        try:
            db_result = asyncio.run(engine.validate_safety_profile(email, herb_name))
            db_result_dict = db_result.to_dict()
        except Exception as e:
            logger.error(f"TemporalReasoningEngine error: {e}", exc_info=True)
            db_result_dict = {"error": str(e)}

        is_safe = fast_result.is_safe and db_result_dict.get("is_safe", True)

        return Response(
            {
                "email": email,
                "herb": herb_name,
                "is_safe": is_safe,
                "fast_check": {
                    "is_safe": fast_result.is_safe,
                    "verdict": fast_result.verdict.value,
                    "reason": fast_result.reason,
                    "contraindications": fast_result.contraindications,
                    "washout_days_remaining": fast_result.washout_days_remaining,
                },
                "temporal_check": db_result_dict,
            }
        )

    # =========================================================================
    # GET /api/temporal/allergy_history/
    # =========================================================================

    @action(detail=False, methods=["get"])
    def allergy_history(self, request):
        """
        Retrieve a user's allergy history.

        Query params:
            email (str, required)
        """
        email = request.query_params.get("email")
        user, err = _get_user_or_error(email)
        if err:
            return err

        records = []
        for allergy in AllergyHistory.objects.filter(user=user).order_by("-created_at"):
            records.append(
                {
                    "id": allergy.pk,
                    "allergen": allergy.allergen,
                    "allergen_type": allergy.allergen_type,
                    "severity": allergy.severity,
                    "onset_date": allergy.onset_date.isoformat() if allergy.onset_date else None,
                    "first_reaction_date": allergy.first_reaction_date.isoformat() if allergy.first_reaction_date else None,
                    "symptoms": allergy.symptoms,
                    "testing_method": allergy.testing_method,
                    "confirmed_by_testing": allergy.confirmed_by_testing,
                    "cross_reactive_allergens": allergy.cross_reactive_allergens,
                    "requires_epipen": allergy.requires_epipen(),
                    "is_newly_developed": allergy.is_newly_developed(),
                }
            )

        return Response({"email": email, "count": len(records), "allergies": records})
