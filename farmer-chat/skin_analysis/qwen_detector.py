"""
Qwen 3.5-0.8B Edge Skin Detector for ServVia — D2C Pipeline
Runs locally via Ollama - no internet required.

Architecture: Describe-then-Classify (D2C)
  Stage 1 — Qwen extracts visual features from the image (what it's good at)
  Stage 2 — Python symptom matcher maps those features to a diagnosis (deterministic)

This separation dramatically improves accuracy on a small model because:
  - 0.8B models are reliable at "what colour/texture/size is this"
  - They are unreliable at "what medical condition is this"
  - The classification logic is medically informed and fully auditable
"""
import os
import re
import base64
import json
import io
import logging
import requests
from PIL import Image

from .disease_detector import SERVVIA_SKIN_CONDITIONS, CONDITION_REMEDIES

logger = logging.getLogger(__name__)

QWEN_MODEL = os.getenv("QWEN_MODEL_TAG", "moondream:latest")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "45"))
EDGE_IMAGE_MAX_DIM = int(os.getenv("EDGE_IMAGE_MAX_DIM", "512"))


# ---------------------------------------------------------------------------
# Condition feature profiles for the symptom matcher (Stage 2)
# Each key maps to a condition in SERVVIA_SKIN_CONDITIONS.
# Values are dicts of feature -> expected value(s).
# Matching features add to the score; mismatches on REQUIRED keys penalise it.
# ---------------------------------------------------------------------------
# Profile structure per condition:
#   "positives": features that add to the score when present  {feature: ([values], weight)}
#   "negatives": features that subtract from the score when present  {feature: ([values], penalty)}
#
# Negative scoring is the key accuracy improvement — absence of a required feature
# now actively reduces the score rather than just not adding to it.

CONDITION_PROFILES = {
    "Acne": {
        "positives": {
            "lesion_type":   (["pustule", "papule", "nodule", "comedone"], 3),
            "affected_area": (["face", "back", "chest", "forehead", "chin", "cheek", "shoulder"], 2),
            "redness":       (["high", "medium"], 1),
            "distribution":  (["scattered", "clustered"], 1),
            "lesion_count":  (["moderate", "many"], 1),
        },
        "negatives": {
            "scaling":       (["thick", "silvery"], 3),   # thick scaling = not acne
            "affected_area": (["foot", "toe", "scalp", "lip", "groin"], 3),
            "lesion_type":   (["welt", "patch"], 3),
        },
    },
    "Eczema (Atopic Dermatitis)": {
        "positives": {
            "texture":       (["rough", "flaky", "dry", "scaly"], 3),
            "scaling":       (["fine", "thin"], 2),
            "border":        (["diffuse"], 2),
            "redness":       (["medium", "high"], 1),
        },
        "negatives": {
            "scaling":       (["thick", "silvery"], 3),   # thick silvery = psoriasis
            "border":        (["sharp", "circular"], 2),  # sharp ring = fungal
            "lesion_type":   (["welt", "wheal"], 2),
        },
    },
    "Psoriasis (mild forms)": {
        "positives": {
            "scaling":       (["thick", "silvery"], 4),
            "border":        (["sharp"], 3),
            "lesion_type":   (["plaque"], 2),
            "texture":       (["rough", "scaly"], 1),
        },
        "negatives": {
            "scaling":       (["none", "fine"], 4),   # no scaling = definitely not psoriasis
            "lesion_type":   (["pustule", "vesicle", "welt"], 2),
        },
    },
    "Heat Rash (Prickly Heat)": {
        "positives": {
            "lesion_size":   (["pinpoint"], 4),
            "distribution":  (["uniform", "dense"], 2),
            "lesion_count":  (["many"], 2),
            "redness":       (["medium", "high"], 1),
        },
        "negatives": {
            "lesion_size":   (["large"], 4),          # large lesions = not heat rash
            "scaling":       (["thick", "silvery"], 2),
            "lesion_type":   (["plaque", "welt"], 2),
        },
    },
    "Hives (Urticaria)": {
        "positives": {
            "lesion_type":   (["welt", "wheal", "patch", "raised patch"], 3),
            "texture":       (["smooth", "raised"], 2),
            "redness":       (["high", "medium"], 1),
            "lesion_count":  (["few", "moderate"], 1),
        },
        "negatives": {
            "scaling":       (["thick", "silvery", "fine"], 2),
            "lesion_size":   (["pinpoint"], 3),         # pinpoint uniform = heat rash
            "distribution":  (["uniform", "dense"], 2),
            "lesion_type":   (["papule", "pustule", "comedone", "crust"], 4),
        },
    },
    "Fungal Infections (Ringworm, Athlete's Foot)": {
        "positives": {
            "border":        (["sharp", "circular", "ring"], 4),
            "scaling":       (["fine", "thin"], 2),
            "texture":       (["scaly", "flaky"], 1),
        },
        "negatives": {
            "border":        (["diffuse"], 3),
            "lesion_type":   (["pustule", "welt", "plaque"], 2),
        },
    },
    "Dry Skin (Xerosis)": {
        "positives": {
            "texture":       (["dry", "rough", "flaky", "tight", "cracked"], 3),
            "scaling":       (["fine", "thin"], 2),
            "redness":       (["none", "low"], 1),
        },
        "negatives": {
            "lesion_type":   (["pustule", "welt", "vesicle", "plaque"], 3),
            "redness":       (["high"], 2),
        },
    },
    "Sunburn": {
        "positives": {
            "redness":       (["high"], 4),
            "texture":       (["smooth", "tender", "uniform"], 2),
            "distribution":  (["uniform", "widespread"], 2),
        },
        "negatives": {
            "lesion_type":   (["pustule", "plaque", "welt", "wart"], 3),
            "scaling":       (["thick", "silvery"], 2),
            "border":        (["circular"], 2),
        },
    },
    "Contact Dermatitis": {
        "positives": {
            "border":        (["sharp", "geometric", "defined"], 4),
            "redness":       (["high", "medium"], 2),
            "affected_area": (["hand", "wrist", "neck", "face", "arm", "wrist"], 2),
        },
        "negatives": {
            "border":        (["diffuse", "circular"], 2),
            "scaling":       (["thick", "silvery"], 2),
        },
    },
    "Dandruff (Seborrheic Dermatitis)": {
        "positives": {
            "affected_area": (["scalp", "head", "hair", "eyebrow"], 4),
            "scaling":       (["fine", "white", "yellow", "flaky"], 3),
            "texture":       (["flaky"], 2),
        },
        "negatives": {
            "affected_area": (["foot", "hand", "arm", "leg", "torso", "chest"], 4),
            "scaling":       (["silvery", "thick"], 3),
            "lesion_type":   (["plaque"], 3),
            "border":        (["sharp"], 2),
        },
    },
    "Razor Bumps (Pseudofolliculitis Barbae)": {
        "positives": {
            "affected_area": (["beard", "neck", "chin", "jawline", "shaved"], 4),
            "lesion_type":   (["papule", "pustule", "bump"], 2),
            "distribution":  (["clustered", "linear"], 1),
        },
        "negatives": {
            "affected_area": (["foot", "scalp", "torso", "back", "chest"], 3),
            "lesion_size":   (["large"], 2),
        },
    },
    "Ingrown Hairs": {
        "positives": {
            "lesion_type":   (["papule", "bump", "pustule"], 2),
            "affected_area": (["leg", "armpit", "groin", "beard", "bikini", "thigh"], 2),
            "distribution":  (["scattered", "few"], 1),
        },
        "negatives": {
            "lesion_count":  (["many"], 2),
            "distribution":  (["uniform", "dense"], 2),
        },
    },
    "Warts (Common Warts)": {
        "positives": {
            "texture":       (["rough", "bumpy", "cauliflower", "raised"], 4),
            "border":        (["sharp"], 2),
            "lesion_type":   (["wart", "nodule", "rough bump", "bump"], 2),
        },
        "negatives": {
            "redness":       (["high"], 2),
            "scaling":       (["silvery"], 2),
            "distribution":  (["uniform", "widespread"], 2),
        },
    },
    "Athlete's Foot": {
        "positives": {
            "affected_area": (["foot", "toe", "sole", "between toes"], 5),
            "texture":       (["scaly", "peeling", "flaky", "soggy", "cracked"], 3),
            "scaling":       (["fine", "thick"], 2),
        },
        "negatives": {
            "affected_area": (["face", "arm", "chest", "back", "scalp", "torso"], 5),
        },
    },
    "Scalp Folliculitis": {
        "positives": {
            "affected_area": (["scalp", "head", "hairline"], 5),
            "lesion_type":   (["pustule", "papule", "bump"], 2),
        },
        "negatives": {
            "affected_area": (["face", "arm", "leg", "torso", "foot"], 4),
        },
    },
    "Mosquito Bite Reactions": {
        "positives": {
            "lesion_count":  (["few"], 3),
            "lesion_type":   (["welt", "bump", "papule"], 2),
            "affected_area": (["arm", "leg", "ankle", "neck", "exposed"], 2),
            "distribution":  (["scattered", "random"], 2),
        },
        "negatives": {
            "lesion_count":  (["many"], 4),             # many lesions = not mosquito bites
            "distribution":  (["uniform", "dense"], 3),
            "scaling":       (["fine", "thick", "silvery"], 2),
        },
    },
    "Cold Sores": {
        "positives": {
            "affected_area": (["lip", "mouth", "perioral"], 5),
            "lesion_type":   (["blister", "vesicle", "ulcer", "crust", "cluster"], 3),
            "distribution":  (["clustered", "single"], 1),
        },
        "negatives": {
            "affected_area": (["back", "chest", "arm", "leg", "torso", "foot", "scalp"], 4),
        },
    },
    "Chapped Lips": {
        "positives": {
            "affected_area": (["lip", "mouth"], 5),
            "texture":       (["dry", "flaky", "cracked", "peeling"], 3),
        },
        "negatives": {
            "affected_area": (["back", "chest", "arm", "leg", "torso", "foot", "scalp"], 5),
            "lesion_type":   (["pustule", "welt", "plaque"], 3),
        },
    },
    "Jock Itch": {
        "positives": {
            "affected_area": (["groin", "inner thigh", "buttock", "genital"], 5),
            "border":        (["sharp", "circular"], 2),
            "redness":       (["high", "medium"], 1),
        },
        "negatives": {
            "affected_area": (["face", "arm", "chest", "back", "scalp", "foot", "torso"], 4),
        },
    },
    "Allergic Rash (Mild Allergic Dermatitis)": {
        "positives": {
            "distribution":  (["scattered", "widespread"], 2),
            "redness":       (["high", "medium"], 2),
            "texture":       (["bumpy", "raised"], 2),
            "border":        (["diffuse"], 1),
        },
        "negatives": {
            "scaling":       (["thick", "silvery"], 2),
            "border":        (["circular", "geometric"], 2),
        },
    },
    "Normal Skin": {
        "positives": {
            "redness":       (["none", "low"], 2),
            "lesion_type":   (["none"], 2),
            "scaling":       (["none"], 2),
            "texture":       (["smooth"], 2),
        },
        "negatives": {
            "redness":       (["high"], 3),
            "lesion_type":   (["pustule", "plaque", "vesicle", "welt", "wart"], 3),
        },
    },
}

# Hard location rules — body part alone is enough to strongly favour or block certain conditions.
# These run after profile scoring and can override a close result.
LOCATION_RULES = {
    "foot":         {"boost": "Athlete's Foot",                              "block": ["Dandruff (Seborrheic Dermatitis)", "Scalp Folliculitis", "Chapped Lips", "Cold Sores"]},
    "toe":          {"boost": "Athlete's Foot",                              "block": ["Dandruff (Seborrheic Dermatitis)", "Scalp Folliculitis"]},
    "scalp":        {"boost": "Dandruff (Seborrheic Dermatitis)", "boost_value": 0.20, "requires": {"scaling": ["fine"], "texture": ["flaky", "scaly", "dry"]}, "block": ["Athlete's Foot", "Chapped Lips", "Jock Itch"]},
    "lip":          {"boost": "Chapped Lips", "requires": {"texture": ["dry", "flaky"]}, "block": ["Athlete's Foot", "Dandruff (Seborrheic Dermatitis)", "Jock Itch", "Scalp Folliculitis"]},
    "mouth":        {"boost": "Cold Sores", "requires": {"lesion_type": ["vesicle", "crust"]}, "block": ["Athlete's Foot", "Dandruff (Seborrheic Dermatitis)", "Jock Itch"]},
    "groin":        {"boost": "Jock Itch",                                   "block": ["Dandruff (Seborrheic Dermatitis)", "Chapped Lips", "Cold Sores", "Scalp Folliculitis"]},
    "beard":        {"boost": "Razor Bumps (Pseudofolliculitis Barbae)",     "block": ["Athlete's Foot", "Jock Itch", "Dandruff (Seborrheic Dermatitis)"]},
    "jawline":      {"boost": "Razor Bumps (Pseudofolliculitis Barbae)",     "block": ["Athlete's Foot", "Jock Itch"]},
}


CONDITION_GUARDRAILS = {
    "Acne": {
        "must_any": {
            "lesion_type": ["papule", "pustule", "bump"],
            "affected_area": ["face", "forehead", "cheek", "chin", "back", "chest", "shoulder"],
        },
        "forbid": {
            "lesion_type": ["welt", "patch"],
            "scaling": ["silvery"],
        },
        "missing_penalty": 0.20,
        "forbid_penalty": 0.35,
    },
    "Hives (Urticaria)": {
        "must_any": {
            "lesion_type": ["welt", "patch"],
            "texture": ["raised", "smooth"],
        },
        "forbid": {
            "lesion_type": ["papule", "pustule", "crust"],
            "scaling": ["thick", "silvery", "fine"],
        },
        "missing_penalty": 0.25,
        "forbid_penalty": 0.35,
    },
    "Psoriasis (mild forms)": {
        "must_any": {
            "scaling": ["thick", "silvery"],
            "lesion_type": ["plaque"],
        },
        "forbid": {
            "lesion_type": ["pustule", "vesicle", "welt"],
            "scaling": ["none"],
        },
        "missing_penalty": 0.25,
        "forbid_penalty": 0.30,
    },
    "Dandruff (Seborrheic Dermatitis)": {
        "must_all": {
            "affected_area": ["scalp", "head", "hairline", "eyebrow"],
        },
        "must_any": {
            "scaling": ["fine"],
            "texture": ["flaky", "dry", "scaly"],
        },
        "forbid": {
            "lesion_type": ["plaque", "welt", "wart"],
            "scaling": ["silvery", "thick"],
            "border": ["sharp"],
        },
        "missing_penalty": 0.28,
        "forbid_penalty": 0.35,
    },
    "Eczema (Atopic Dermatitis)": {
        "must_any": {
            "texture": ["dry", "flaky", "scaly", "rough"],
            "border": ["diffuse"],
        },
        "forbid": {
            "scaling": ["silvery"],
            "border": ["circular"],
        },
        "missing_penalty": 0.18,
        "forbid_penalty": 0.25,
    },
    "Fungal Infections (Ringworm, Athlete's Foot)": {
        "must_any": {
            "border": ["circular", "sharp"],
            "scaling": ["fine", "thick"],
        },
        "forbid": {
            "lesion_type": ["welt", "pustule"],
        },
        "missing_penalty": 0.20,
        "forbid_penalty": 0.20,
    },
    "Heat Rash (Prickly Heat)": {
        "must_all": {
            "lesion_size": ["pinpoint"],
            "lesion_count": ["many"],
        },
        "must_any": {
            "distribution": ["uniform", "clustered"],
        },
        "forbid": {
            "lesion_type": ["welt", "plaque"],
            "lesion_size": ["large"],
        },
        "missing_penalty": 0.35,
        "forbid_penalty": 0.35,
    },
    "Sunburn": {
        "must_any": {
            "redness": ["high"],
            "distribution": ["uniform"],
        },
        "forbid": {
            "lesion_type": ["pustule", "papule", "welt", "wart"],
        },
        "missing_penalty": 0.20,
        "forbid_penalty": 0.25,
    },
    "Dry Skin (Xerosis)": {
        "must_any": {
            "texture": ["dry", "flaky", "rough"],
            "scaling": ["fine"],
        },
        "forbid": {
            "redness": ["high"],
            "lesion_type": ["pustule", "welt", "vesicle", "plaque"],
        },
        "missing_penalty": 0.15,
        "forbid_penalty": 0.25,
    },
    "Contact Dermatitis": {
        "must_any": {
            "border": ["sharp"],
            "redness": ["high", "medium"],
        },
        "forbid": {
            "scaling": ["silvery"],
        },
        "missing_penalty": 0.18,
        "forbid_penalty": 0.20,
    },
    "Athlete's Foot": {
        "must_all": {
            "affected_area": ["foot", "toe", "sole", "between toes"],
        },
        "must_any": {
            "texture": ["scaly", "flaky", "dry"],
            "scaling": ["fine", "thick"],
        },
        "missing_penalty": 0.30,
        "forbid_penalty": 0.20,
    },
    "Cold Sores": {
        "must_all": {
            "affected_area": ["lip", "mouth", "perioral"],
        },
        "must_any": {
            "lesion_type": ["vesicle", "crust"],
        },
        "missing_penalty": 0.30,
        "forbid_penalty": 0.20,
    },
    "Chapped Lips": {
        "must_all": {
            "affected_area": ["lip", "mouth"],
        },
        "must_any": {
            "texture": ["dry", "flaky", "rough"],
        },
        "forbid": {
            "lesion_type": ["vesicle", "crust", "pustule"],
        },
        "missing_penalty": 0.25,
        "forbid_penalty": 0.25,
    },
}


def _matches_expected(actual: str, expected_values: list[str]) -> bool:
    if not actual:
        return False
    return any(ev in actual or actual in ev for ev in expected_values)


def _apply_guardrails(scores: dict, features: dict) -> dict:
    """Condition-level constraints to avoid location-only or contradictory predictions."""
    adjusted = dict(scores)

    for condition, rules in CONDITION_GUARDRAILS.items():
        if condition not in adjusted:
            continue

        missing_penalty = float(rules.get("missing_penalty", 0.2))
        forbid_penalty = float(rules.get("forbid_penalty", 0.25))

        for feat, expected in rules.get("must_all", {}).items():
            actual = str(features.get(feat, ""))
            if not _matches_expected(actual, expected):
                adjusted[condition] -= missing_penalty

        must_any = rules.get("must_any", {})
        if must_any:
            any_hit = False
            for feat, expected in must_any.items():
                actual = str(features.get(feat, ""))
                if _matches_expected(actual, expected):
                    any_hit = True
                    break
            if not any_hit:
                adjusted[condition] -= missing_penalty

        for feat, forbidden in rules.get("forbid", {}).items():
            actual = str(features.get(feat, ""))
            if _matches_expected(actual, forbidden):
                adjusted[condition] -= forbid_penalty

    return adjusted


def _normalize_features(features: dict) -> dict:
    """Map noisy model outputs into canonical buckets used by scoring rules."""
    norm = {k: (str(v).strip().lower() if v is not None else "") for k, v in features.items()}

    def _map_value(key: str, aliases: dict[str, list[str]], default: str = "") -> str:
        raw = norm.get(key, "")
        if not raw:
            return default
        for canonical, words in aliases.items():
            if any(w in raw for w in words):
                return canonical
        return raw

    norm["redness"] = _map_value(
        "redness",
        {
            "none": ["none", "normal", "no redness", "not red"],
            "low": ["mild", "slight", "faint", "low"],
            "medium": ["moderate", "medium"],
            "high": ["high", "marked", "severe", "intense", "bright red"],
        },
        default="none",
    )

    norm["scaling"] = _map_value(
        "scaling",
        {
            "none": ["none", "no scale", "smooth"],
            "fine": ["fine", "thin", "mild", "light", "powdery", "flaky"],
            "thick": ["thick", "heavy", "coarse"],
            "silvery": ["silvery", "silver", "silvery-white", "micaceous"],
        },
        default="none",
    )

    norm["lesion_type"] = _map_value(
        "lesion_type",
        {
            "none": ["none", "no lesion"],
            "papule": ["papule", "papules", "small bump", "bumps"],
            "pustule": ["pustule", "pustules", "pus"],
            "plaque": ["plaque", "plaque-like"],
            "vesicle": ["vesicle", "blister", "blisters"],
            "welt": ["welt", "wheal", "hive"],
            "wart": ["wart", "verruca"],
            "patch": ["patch", "patches"],
            "bump": ["bump", "nodular", "nodule"],
            "crust": ["crust", "scab"],
        },
        default="none",
    )

    norm["lesion_size"] = _map_value(
        "lesion_size",
        {
            "pinpoint": ["pinpoint", "1mm", "tiny", "very small", "minute"],
            "small": ["small", "2mm", "3mm", "few mm"],
            "large": ["large", "big", "5mm", "10mm", "raised plaque", "welts"],
        },
        default="small",
    )

    norm["lesion_count"] = _map_value(
        "lesion_count",
        {
            "none": ["none", "single smooth", "no lesions"],
            "few": ["few", "sparse", "isolated"],
            "moderate": ["moderate", "several"],
            "many": ["many", "numerous", "multiple", "dense", "widespread"],
        },
        default="moderate",
    )

    norm["distribution"] = _map_value(
        "distribution",
        {
            "single": ["single", "localized"],
            "scattered": ["scattered", "random"],
            "clustered": ["clustered", "grouped"],
            "uniform": ["uniform", "even", "dense"],
            "linear": ["linear", "line"],
        },
        default="scattered",
    )

    norm["texture"] = _map_value(
        "texture",
        {
            "smooth": ["smooth", "flat"],
            "rough": ["rough", "coarse"],
            "flaky": ["flaky", "peeling"],
            "scaly": ["scaly", "scale"],
            "bumpy": ["bumpy", "nodular", "uneven"],
            "dry": ["dry", "xerotic"],
            "raised": ["raised", "elevated"],
        },
        default="smooth",
    )

    norm["border"] = _map_value(
        "border",
        {
            "none": ["none", "not clear"],
            "sharp": ["sharp", "defined", "well-defined"],
            "diffuse": ["diffuse", "poorly defined", "ill-defined"],
            "circular": ["circular", "ring", "annular"],
        },
        default="diffuse",
    )

    return norm


def _apply_differential_rules(features: dict, condition: str, confidence: int) -> tuple[str, int]:
    """Hard rules for commonly confused condition pairs."""
    area = features.get("affected_area", "")
    lesion_type = features.get("lesion_type", "")
    lesion_size = features.get("lesion_size", "")
    lesion_count = features.get("lesion_count", "")
    scaling = features.get("scaling", "")
    border = features.get("border", "")
    texture = features.get("texture", "")
    distribution = features.get("distribution", "")

    # Acne should beat hives when morphology is papulopustular in classic acne zones.
    if lesion_type in ("papule", "pustule") and any(k in area for k in ("face", "forehead", "chin", "cheek", "back", "chest")):
        if scaling not in ("thick", "silvery"):
            return "Acne", max(confidence, 80)

    # Foot lesions with scaling should strongly prefer tinea pedis.
    if any(k in area for k in ("foot", "toe", "sole", "between toes")) and scaling in ("fine", "thick"):
        return "Athlete's Foot", max(confidence, 80)

    # Scalp: flaky scaling tends to dandruff; follicular bumps tend to folliculitis.
    if "scalp" in area or "hairline" in area:
        if scaling in ("silvery", "thick") or lesion_type == "plaque":
            return "Psoriasis (mild forms)", max(confidence, 84)
        if scaling == "fine" and texture in ("flaky", "scaly", "dry"):
            return "Dandruff (Seborrheic Dermatitis)", max(confidence, 78)
        if lesion_type in ("papule", "pustule", "bump"):
            return "Scalp Folliculitis", max(confidence, 78)

    # Lip/perioral differential.
    if any(k in area for k in ("lip", "mouth", "perioral")):
        if lesion_type in ("vesicle", "crust"):
            return "Cold Sores", max(confidence, 80)
        if texture in ("dry", "flaky") and lesion_type not in ("vesicle", "crust"):
            return "Chapped Lips", max(confidence, 78)

    # Heat rash vs hives.
    if lesion_size == "pinpoint" and lesion_count == "many" and distribution == "uniform":
        return "Heat Rash (Prickly Heat)", max(confidence, 82)
    if lesion_type in ("welt", "patch") or (lesion_size == "large" and lesion_count in ("few", "moderate") and texture == "smooth"):
        return "Hives (Urticaria)", max(confidence, 80)

    # Eczema vs psoriasis.
    if scaling == "silvery" and border == "sharp":
        return "Psoriasis (mild forms)", max(confidence, 84)
    if texture in ("dry", "flaky", "scaly") and border == "diffuse" and scaling in ("fine", "none"):
        return "Eczema (Atopic Dermatitis)", max(confidence, 76)

    # Ring-like lesions with scale should prefer fungal diagnosis.
    if border == "circular" and scaling in ("fine", "thick"):
        return "Fungal Infections (Ringworm, Athlete's Foot)", max(confidence, 80)

    return condition, confidence


def _apply_confidence_safety(features: dict, condition: str, confidence: int, margin: float) -> int:
    """Cap confidence when hallmark evidence for a condition is missing."""
    lesion_type = features.get("lesion_type", "")
    scaling = features.get("scaling", "")
    border = features.get("border", "")
    texture = features.get("texture", "")
    area = features.get("affected_area", "")
    lesion_size = features.get("lesion_size", "")
    lesion_count = features.get("lesion_count", "")

    if condition == "Hives (Urticaria)":
        if lesion_type not in ("welt", "patch"):
            confidence = min(confidence, 60)

    if condition == "Acne":
        if lesion_type not in ("papule", "pustule", "bump"):
            confidence = min(confidence, 60)

    if condition == "Psoriasis (mild forms)":
        if scaling not in ("thick", "silvery") and lesion_type != "plaque":
            confidence = min(confidence, 58)

    if condition == "Dandruff (Seborrheic Dermatitis)":
        if "scalp" not in area and "hair" not in area:
            confidence = min(confidence, 55)
        if scaling not in ("fine",):
            confidence = min(confidence, 58)

    if condition == "Heat Rash (Prickly Heat)":
        if not (lesion_size == "pinpoint" and lesion_count == "many"):
            confidence = min(confidence, 58)

    if condition == "Fungal Infections (Ringworm, Athlete's Foot)":
        if border not in ("circular", "sharp"):
            confidence = min(confidence, 60)

    # Global uncertainty clamp
    if margin < 0.06 and condition != "Normal Skin":
        confidence = min(confidence, 62)

    return confidence


def _build_reasoning(features: dict, condition: str) -> str:
    """Generate a human-readable explanation of why a condition was selected."""
    area = features.get("affected_area", "the affected area")
    description = features.get("description", "")
    redness = features.get("redness", "")
    scaling = features.get("scaling", "none")
    lesion_type = features.get("lesion_type", "")
    lesion_size = features.get("lesion_size", "")
    lesion_count = features.get("lesion_count", "")
    texture = features.get("texture", "")
    border = features.get("border", "")
    distribution = features.get("distribution", "")

    lines = []

    if description:
        lines.append(description)

    observations = []
    if redness and redness != "none":
        observations.append(f"{redness} redness")
    if lesion_type and lesion_type != "none":
        size_prefix = f"{lesion_size} " if lesion_size else ""
        count_prefix = f"{lesion_count} " if lesion_count else ""
        observations.append(f"{count_prefix}{size_prefix}{lesion_type}s")
    if texture and texture != "smooth":
        observations.append(f"{texture} texture")
    if scaling and scaling != "none":
        observations.append(f"{scaling} scaling")
    if border and border != "none":
        observations.append(f"{border} borders")
    if distribution and distribution not in ("none", "single"):
        observations.append(f"{distribution} distribution")

    if observations:
        obs_str = ", ".join(observations)
        lines.append(f"The image shows {obs_str} on {area}.")

    # Condition-specific reasoning
    reasoning_map = {
        "Acne": "The combination of pustules or papules on a facial or body area, with redness and no scaling, is characteristic of acne.",
        "Eczema (Atopic Dermatitis)": "The dry, flaky texture with diffuse borders and fine scaling points to eczema rather than a more sharply defined condition.",
        "Psoriasis (mild forms)": "The thick silvery scaling with sharp, well-defined borders is the hallmark of psoriasis plaques.",
        "Heat Rash (Prickly Heat)": "The large number of tiny, uniform pinpoint lesions densely packed together is consistent with blocked sweat ducts seen in heat rash.",
        "Hives (Urticaria)": "The raised, smooth patches with relatively defined edges appearing across the skin are typical of hives, which arise from an allergic or immune response.",
        "Fungal Infections (Ringworm, Athlete's Foot)": "The circular or ring-shaped border with fine scaling is the classic presentation of a fungal infection.",
        "Dry Skin (Xerosis)": "The dry, tight, flaky appearance without significant redness or lesions is consistent with xerosis.",
        "Sunburn": "The uniform high redness over exposed skin without scaling or raised lesions suggests sunburn.",
        "Contact Dermatitis": "The sharply defined, geometric border of redness suggests direct contact with an irritant or allergen.",
        "Dandruff (Seborrheic Dermatitis)": "The white or yellow flaky scaling on the scalp or hairline is characteristic of seborrheic dermatitis.",
        "Razor Bumps (Pseudofolliculitis Barbae)": "The papules clustered along a shaved area such as the neck or jawline suggest ingrown hairs from shaving.",
        "Warts (Common Warts)": "The rough, cauliflower-like surface with sharp borders and raised texture is typical of common warts.",
        "Athlete's Foot": "The peeling, scaly skin between or around the toes is the classic presentation of tinea pedis.",
        "Hives (Urticaria)": "The large smooth patches appearing across the skin with diffuse spread suggest an urticarial reaction.",
        "Mosquito Bite Reactions": "A small number of scattered raised bumps on exposed skin is consistent with insect bite reactions.",
        "Allergic Rash (Mild Allergic Dermatitis)": "The widespread redness and bumpy texture distributed across the skin suggests a mild allergic skin reaction.",
        "Normal Skin": "No significant lesions, redness, or abnormal texture were detected. The skin appears healthy.",
    }

    specific = reasoning_map.get(condition)
    if specific:
        lines.append(specific)

    return " ".join(lines) if lines else f"Visual features on {area} are consistent with {condition}."


def _classify_from_features(features: dict) -> tuple[str, int]:
    """
    Score each condition profile against extracted features.
    Positive feature matches add to the score.
    Negative feature matches subtract from the score.
    Location rules then boost or block conditions based on body part.
    Returns (condition_name, confidence_percent).
    """
    normalized = _normalize_features(features)
    area = normalized.get("affected_area", "").lower()
    scores = {}

    for condition, profile in CONDITION_PROFILES.items():
        positives = profile.get("positives", {})
        negatives = profile.get("negatives", {})

        max_possible = sum(w for _, (_, w) in positives.items())
        score = 0
        matched_positives = 0

        for feat, (expected, weight) in positives.items():
            actual = normalized.get(feat, "").lower().strip()
            if actual and any(ev.lower() in actual or actual in ev.lower() for ev in expected):
                score += weight
                matched_positives += 1

        for feat, (penalise_vals, penalty) in negatives.items():
            actual = normalized.get(feat, "").lower().strip()
            if actual and any(pv.lower() in actual or actual in pv.lower() for pv in penalise_vals):
                score -= penalty

        coverage_bonus = 0.05 * matched_positives
        scores[condition] = (score / max_possible + coverage_bonus) if max_possible > 0 else 0

    # Apply guardrails before location boosts.
    scores = _apply_guardrails(scores, normalized)

    # Apply location rules
    blocked = set()
    for keyword, rule in LOCATION_RULES.items():
        if keyword in area:
            boost_condition = rule.get("boost")
            requires = rule.get("requires", {})
            requirements_pass = True
            for req_feat, req_vals in requires.items():
                req_actual = normalized.get(req_feat, "")
                if not _matches_expected(req_actual, req_vals):
                    requirements_pass = False
                    break

            if boost_condition and boost_condition in scores and requirements_pass:
                boost_value = float(rule.get("boost_value", 0.4))
                scores[boost_condition] = min(scores[boost_condition] + boost_value, 1.0)
                logger.info(f"[D2C] Location boost: '{keyword}' in area -> +{boost_value} to {boost_condition}")
            elif boost_condition and boost_condition in scores and requires:
                logger.info(f"[D2C] Location boost skipped for {boost_condition}: required morphology not met")
            for b in rule.get("block", []):
                blocked.add(b)
                logger.info(f"[D2C] Location block: '{keyword}' in area -> blocked {b}")

    for b in blocked:
        scores[b] = -99

    ranked = sorted(((c, s) for c, s in scores.items() if s > -99), key=lambda x: x[1], reverse=True)
    best_condition, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = max(best_score - second_score, 0.0)

    top3 = ranked[:3]
    logger.info(f"[D2C] Top 3 scores: {top3}")

    # Confidence includes class separation to reduce false high-confidence outputs.
    confidence = int(35 + max(best_score, 0) * 42 + margin * 30)
    confidence = min(confidence, 95)

    # Apply hard differential rules after profile scoring.
    best_condition, confidence = _apply_differential_rules(normalized, best_condition, confidence)
    confidence = _apply_confidence_safety(normalized, best_condition, confidence, margin)

    if best_score < 0.25:
        best_condition = "Normal Skin"
        confidence = 52

    # If classifier is uncertain between top candidates, force low confidence so router escalates.
    if margin < 0.08 and best_condition != "Normal Skin":
        confidence = min(confidence, 64)

    return best_condition, confidence


def is_ollama_running() -> bool:
    """
    Check if Ollama is reachable, the configured model is available,
    and that model has a vision encoder (projector_info not None).
    Text-only models like qwen3.5:0.8b are rejected so the router
    falls through to Gemini automatically.
    """
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return False
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(QWEN_MODEL in m for m in models)
        if not available:
            logger.warning(
                f"Edge vision model '{QWEN_MODEL}' not found. "
                f"Run: ollama pull {QWEN_MODEL}"
            )
            return False
        # Verify the model actually has a vision encoder
        show_resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/show",
            json={"name": QWEN_MODEL},
            timeout=5,
        )
        if show_resp.status_code == 200:
            projector = show_resp.json().get("projector_info")
            if not projector:
                logger.warning(
                    f"Model '{QWEN_MODEL}' has no vision encoder — "
                    "routing to Gemini. Pull a vision model (e.g. moondream) "
                    "for local inference."
                )
                return False
        return True
    except Exception as e:
        logger.warning(f"Ollama not reachable: {e}")
        return False


def _prepare_image_b64(image_path: str, max_dim: int = EDGE_IMAGE_MAX_DIM) -> str:
    """Resize image to max_dim on its longest side, then base64-encode as JPEG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"[Edge AI] Image resized {w}x{h} -> {new_w}x{new_h}")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_ollama(prompt: str, image_b64: str) -> str:
    """Send a prompt + image to Ollama and return the raw content string."""
    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 512},
    }
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()

    message = resp.json().get("message", {})
    content = message.get("content", "").strip()
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    if not content:
        thinking = message.get("thinking", "")
        if thinking:
            last = thinking.rfind("}")
            first = thinking.rfind("{", 0, last)
            if first != -1 and last != -1:
                content = thinking[first:last + 1].strip()

    return content


def detect_skin_disease_qwen(image_path: str) -> dict:
    """
    D2C pipeline:
      Stage 1 — Qwen extracts 8 visual features from the image
      Stage 2 — Python symptom matcher produces the final diagnosis
    """
    if not is_ollama_running():
        return {
            "success": False,
            "error": f"Ollama not running or model '{QWEN_MODEL}' not pulled.",
            "error_type": "ollama_unavailable",
        }

    try:
        image_b64 = _prepare_image_b64(image_path)

        # ------------------------------------------------------------------
        # Stage 1: Feature extraction
        # Ask Qwen only about what it can see, not what it should diagnose.
        # Shorter prompt = faster inference.
        # ------------------------------------------------------------------
        feature_prompt = """Look at this skin image and describe only what you visually observe.
    Do NOT diagnose.

    Visual guidance for consistency:
    - Use lesion_type=papule or pustule for acne-like follicular bumps.
    - Use lesion_type=welt or patch only for smooth raised wheals/hives.
    - Use lesion_type=plaque with scaling=silvery or thick for psoriasis-like plaques.
    - Use scaling=fine and texture=flaky for dandruff-like scalp flaking.
    - Use lesion_size=pinpoint and lesion_count=many for heat-rash-like tiny dense lesions.
    - Use border=circular when a ring-like edge is visible.

Return ONLY a JSON object with these exact keys:

{
  "redness": "none OR low OR medium OR high",
  "scaling": "none OR fine OR thick OR silvery",
  "lesion_type": "none OR papule OR pustule OR plaque OR vesicle OR welt OR wart OR patch OR bump OR crust",
  "lesion_size": "pinpoint OR small OR large",
  "lesion_count": "none OR few OR moderate OR many",
  "distribution": "single OR scattered OR clustered OR uniform OR linear",
  "texture": "smooth OR rough OR flaky OR scaly OR bumpy OR dry OR raised",
  "border": "none OR sharp OR diffuse OR circular",
  "affected_area": "describe the body part you can see",
  "description": "one sentence describing what you see"
}

Only describe what is visible. Return raw JSON only."""

        logger.info(f"[D2C] Stage 1: Extracting visual features via Qwen...")
        raw = _call_ollama(feature_prompt, image_b64)
        logger.info(f"[D2C] Feature extraction response length: {len(raw)}")

        # Parse features
        try:
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            elif "{" in raw and "}" in raw:
                raw = raw[raw.find("{"):raw.rfind("}") + 1]
            features = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"[D2C] Feature JSON parse failed: {e} | Raw: {raw[:200]}")
            return {"success": False, "error": f"Feature extraction failed: {e}"}

        logger.info(f"[D2C] Extracted features: {features}")

        # ------------------------------------------------------------------
        # Stage 2: Classify using symptom profile matcher
        # ------------------------------------------------------------------
        logger.info("[D2C] Stage 2: Running symptom profile matcher...")
        detected_condition, confidence = _classify_from_features(features)

        # Validate against supported list
        valid = [c.lower() for c in SERVVIA_SKIN_CONDITIONS]
        if detected_condition.lower() not in valid:
            detected_condition = "Normal Skin"
            confidence = 55

        logger.info(f"[D2C] Final diagnosis: {detected_condition} ({confidence}%)")

        # Severity: derive from confidence and lesion characteristics
        redness = features.get("redness", "none").lower()
        lesion_count = features.get("lesion_count", "none").lower()
        if confidence >= 80 and redness == "high":
            severity_raw = "moderate"
        elif confidence >= 85 and lesion_count == "many":
            severity_raw = "moderate"
        elif detected_condition == "Normal Skin":
            severity_raw = "normal"
        else:
            severity_raw = "mild"

        severity_map = {
            "severe":   ("Severe",   "High",     "URGENT: This condition requires immediate medical attention."),
            "moderate": ("Moderate", "Moderate", "Try home remedies, but consult a doctor if symptoms persist."),
            "mild":     ("Mild",     "Low",      "This can typically be managed with home remedies."),
            "normal":   ("Normal",   "None",     "Your skin appears healthy."),
        }
        severity, urgency, urgency_note = severity_map.get(severity_raw, ("Mild", "Low", "Try home remedies and monitor progress."))

        remedies = CONDITION_REMEDIES.get(detected_condition, [
            "Consult a healthcare professional for accurate diagnosis",
            "Maintain good hygiene",
            "Keep affected area clean and dry",
        ])

        return {
            "success": True,
            "disease": detected_condition,
            "confidence": "Very High" if confidence > 90 else "High" if confidence > 75 else "Moderate" if confidence > 60 else "Low",
            "confidence_score": confidence / 100,
            "confidence_numeric": confidence,
            "severity": severity,
            "urgency": urgency,
            "description": features.get("description", ""),
            "key_features": [
                f"Redness: {features.get('redness', 'N/A')}",
                f"Lesion type: {features.get('lesion_type', 'N/A')}",
                f"Texture: {features.get('texture', 'N/A')}",
                f"Scaling: {features.get('scaling', 'N/A')}",
            ],
            "affected_area": features.get("affected_area", "Not specified"),
            "differential_diagnosis": [],
            "distinguishing_features": f"Key indicators: {features.get('lesion_type', 'no specific lesion')} with {features.get('texture', 'unremarkable')} texture, {features.get('scaling', 'no')} scaling, and {features.get('border', 'undefined')} borders.",
            "visual_analysis": {
                "border_type": features.get("border", ""),
                "scale_type": features.get("scaling", ""),
                "texture": features.get("texture", ""),
                "lesion_size": features.get("lesion_size", ""),
                "lesion_count": features.get("lesion_count", ""),
            },
            "reasoning": _build_reasoning(features, detected_condition),
            "recommendations": remedies,
            "urgency_note": urgency_note,
            "inference_source": "edge",
            "analyzed_by": f"qwen-d2c ({QWEN_MODEL})",
        }

    except requests.Timeout:
        logger.warning(f"[Edge AI] Qwen timed out after {OLLAMA_TIMEOUT}s")
        return {"success": False, "error": "Qwen inference timed out", "error_type": "timeout"}
    except Exception as e:
        logger.error(f"[Edge AI] Unexpected error: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
