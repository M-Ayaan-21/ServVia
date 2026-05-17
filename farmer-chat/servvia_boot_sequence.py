#!/usr/bin/env python3
"""Hacker-style terminal boot sequence for the ServVia stack."""

import sys
import time


RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
BLACK_BACKGROUND = "\033[40m"
RED_ON_BLACK = f"{RED}{BLACK_BACKGROUND}"
CLEAR_SCREEN = "\033[2J\033[3J\033[H"

PROJECT_NAME = "ServVia"


def slow_type(text: str, delay: float = 0.05, color: str = RED_ON_BLACK, end: str = "\n") -> None:
    """Write text one character at a time to simulate human terminal input."""
    sys.stdout.write(color)
    sys.stdout.flush()
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write(f"{RESET}{end}")
    sys.stdout.flush()


def fast_boot(lines: list[str], delay: float = 0.035) -> None:
    """Print rapid-fire initialization messages."""
    for line in lines:
        sys.stdout.write(f"{DIM}{RED_ON_BLACK}{line}{RESET}{BLACK_BACKGROUND}\n")
        sys.stdout.flush()
        time.sleep(delay)


def main() -> None:
    sys.stdout.write(f"{BLACK_BACKGROUND}{CLEAR_SCREEN}")
    sys.stdout.flush()

    slow_type(f"> init {PROJECT_NAME}", delay=0.05)
    time.sleep(0.25)
    slow_type("EXECUTING INITIALIZATION PROTOCOL...", delay=0.02)
    time.sleep(0.2)

    fast_boot(
        [
            "[sys] mounting django_rest_framework gateway...",
            "[sys] binding secure clinical vector retrieval bus...",
            "[rag] hydrating retrieval/rephrase/rerank intelligence chain...",
            "[sec] bypassing unsafe-remedy generation path via deterministic guardrails...",
            "[db] indexing UserProfile, MedicationHistory, SymptomOnset ledgers...",
            "[kg] loading herb -> phytochemical -> target -> disease graph edges...",
            "[llm] negotiating Groq, Gemini, Qwen, translation, ASR, and TTS adapters...",
            "[ui] arming streaming token transport for Vite/React console...",
            "[audit] routing contraindication telemetry into validation logs...",
        ]
    )

    features = [
        "Booting Agentic RAG Orchestrator: clinical vector retrieval fused with intent classification, query rephrasing, LLM reranking, and personalized response synthesis.",
        "Activating Neuro-Symbolic Trust Engine: GRADE-style evidence scoring, PubMed citation awareness, herb-herb interaction maps, and contraindication suppression online.",
        "Loading Temporal Pharmacovigilance Core: real MedicationHistory windows, stabilization checks, washout-period enforcement, symptom acuity analysis, and pre-generation safety blocks.",
        "Compiling Integrative Knowledge Graph: herb nodes linked to phytochemicals, biological targets, disease pathways, clinical trials, dosage metadata, and safety constraints.",
        "Engaging Multi-Agent Verification Graph: Reasoner, Proposer, Critic, Safety Validator, and deterministic fallback loops coordinating through LangGraph.",
        "Deploying Multimodal Diagnostics Layer: Qwen edge skin analysis with describe-then-classify matching plus Gemini lab-report extraction, JSON structuring, and clinical summary generation.",
        "Synchronizing Premium Patient Interface: onboarding medical profiles, streaming chat tokens, voice input/output hooks, trust cards, chronobiology cards, image uploads, and PDF lab ingestion.",
    ]

    sys.stdout.write("\n")
    sys.stdout.flush()
    for feature in features:
        slow_type(f"[✓] {feature}", delay=0.05)
        time.sleep(0.18)

    sys.stdout.write("\n")
    sys.stdout.flush()
    slow_type("SYSTEM READY. Awaiting input...", delay=0.05)


if __name__ == "__main__":
    main()
