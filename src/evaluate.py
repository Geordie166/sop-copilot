"""
evaluate.py — a small evaluation harness for the SOP Copilot RAG pipeline.

WHY THIS MODULE EXISTS:
Unit tests (tests/test_ingest.py, test_embed.py, test_agent.py) check that
individual functions behave correctly in isolation — they don't tell you
whether the *system* actually retrieves the right document for a realistic
question, or whether Claude's final answer is actually grounded and
correctly cited. That's a different kind of question ("is the answer
good?" vs. "does the code work?"), and answering it requires a labeled set
of representative questions to check against — that's what
data/eval/qa_pairs.json is, and this script is what runs the checks.

WHY retrieval and generation are evaluated separately, with generation
opt-in via a flag: retrieval evaluation (does the vector search return the
right source document?) is fully local, deterministic, and free — safe to
run every time. Generation evaluation calls the real Claude API for every
question, which costs money and is non-deterministic, so it's gated behind
an explicit --full flag rather than running by default.

WHY generation results are only heuristically flagged, not strictly
pass/fail graded: judging whether a free-text answer is "good" (concise,
correctly citing sources, appropriately declining out-of-scope questions)
is a genuinely hard automated-grading problem. Rather than pretend a
brittle keyword check is a reliable verdict, this script prints the full
transcript for a human to read, and only uses simple heuristics (does the
expected filename appear in the answer? does a refusal look like a
refusal?) as a rough signal, not a final judgment.
"""

import argparse
import json
import os
from pathlib import Path

from anthropic import Anthropic

import agent
from embed import get_chroma_collection, query_sop_documents

QA_PAIRS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval" / "qa_pairs.json"

# WHY these specific phrases: they're the kind of language the system
# prompt in agent.py asks Claude to use when it can't answer from the
# document library (see SYSTEM_PROMPT rule 3 in agent.py). This is a
# heuristic, not a guarantee — Claude could phrase a refusal differently —
# which is exactly why generation eval results are meant to be read, not
# just tallied.
REFUSAL_PHRASES = [
    "don't have",
    "do not have",
    "no documents",
    "not addressed",
    "not covered",
    "can't answer",
    "cannot answer",
    "unable to find",
    "no relevant",
    "unrelated",
]


def load_qa_pairs() -> list[dict]:
    with open(QA_PAIRS_PATH, encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(collection, qa_pairs: list[dict]) -> None:
    """
    For every QA pair with a known expected source document, check whether
    query_sop_documents() surfaces that document at rank 1 (recall@1) and
    anywhere in the top 3 (recall@3).

    WHY skip should_refuse pairs here: those questions have no correct
    source document by design (expected_source is None) — there's nothing
    for retrieval to be "correct" about, so including them would only
    distort the recall numbers for a metric they aren't meant to test.
    """
    retrieval_cases = [qa for qa in qa_pairs if not qa["should_refuse"]]

    top1_hits = 0
    top3_hits = 0

    print("=== Retrieval evaluation (local, free, deterministic) ===\n")

    for qa in retrieval_cases:
        results = query_sop_documents(collection, qa["question"], n_results=3)
        sources = [r["source"] for r in results]

        top1_hit = sources[:1] == [qa["expected_source"]]
        top3_hit = qa["expected_source"] in sources

        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)

        status = "PASS" if top1_hit else ("PARTIAL" if top3_hit else "FAIL")
        print(f"[{status}] {qa['question']}")
        print(f"  expected: {qa['expected_source']}")
        print(f"  got (ranked): {sources}\n")

    total = len(retrieval_cases)
    print(f"Recall@1: {top1_hits}/{total} ({top1_hits / total:.0%})")
    print(f"Recall@3: {top3_hits}/{total} ({top3_hits / total:.0%})\n")


def evaluate_generation(client: Anthropic, collection, qa_pairs: list[dict]) -> None:
    """
    Run every QA pair through the real agent.ask() (live Claude API calls)
    and print the full transcript alongside a rough heuristic check, for a
    human to read and judge.
    """
    print("=== Generation evaluation (live API calls — costs money) ===\n")

    for qa in qa_pairs:
        answer = agent.ask(client, collection, qa["question"])
        answer_lower = answer.lower()

        if qa["should_refuse"]:
            looks_like_refusal = any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
            heuristic = "looks like a refusal" if looks_like_refusal else "does NOT look like a refusal — review"
        else:
            cites_expected_source = qa["expected_source"].lower() in answer_lower
            heuristic = "cites expected source" if cites_expected_source else "missing expected citation — review"

        print(f"Q: {qa['question']}")
        print(f"Expected: {'refusal' if qa['should_refuse'] else qa['expected_source']}")
        print(f"Heuristic: {heuristic}")
        print(f"A: {answer}\n")
        print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SOP Copilot's retrieval accuracy and (optionally) generated answers."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run generation evaluation against the real Claude API (costs money).",
    )
    args = parser.parse_args()

    qa_pairs = load_qa_pairs()
    collection = get_chroma_collection()

    evaluate_retrieval(collection, qa_pairs)

    if args.full:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        evaluate_generation(client, collection, qa_pairs)
    else:
        print("Skipped generation evaluation (pass --full to run it against the real API).")
