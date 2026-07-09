# SOP Copilot — Roadmap

A running record of what's built, what's in flight, and what's left. See
[README.md](README.md) for architecture and setup; this file tracks
project status over time.

## Completed

**Data**
- 5 synthetic, fictional healthcare call center documents in `data/raw/`:
  3 SOPs (HIPAA identity verification, call escalation, after-call work
  documentation) and 2 RCAs (elevated handle time, repeat-call volume).

**Pipeline**
- `src/ingest.py` — loads `.txt` documents, chunks them with configurable
  word-count size and overlap (default 500/50).
- `src/embed.py` — persistent local ChromaDB store (`data/processed/`),
  embeds chunks with source/chunk-index metadata, exposes a query
  function returning ranked results with relevance distance.
- `src/agent.py` — Claude tool-use loop (`search_sop_documents` tool),
  citation and "don't guess" rules in the system prompt, multi-turn
  conversation support, and both an interactive CLI (`python src/agent.py`)
  and a non-interactive demo mode (`--demo`).

**Quality**
- 15 unit tests across `tests/test_ingest.py`, `test_embed.py`,
  `test_agent.py` — deterministic chunking logic tested directly,
  embedding/retrieval tested against a real local ChromaDB instance,
  and the tool-use loop tested with a mocked Anthropic client.
- `src/evaluate.py` — evaluation harness against a labeled 7-question
  dataset (`data/eval/qa_pairs.json`): free local retrieval recall@1/@3
  (currently 100%), plus an opt-in `--full` mode that runs real Claude
  calls and checks citation/refusal heuristics (currently 5/5 correct
  citations, 2/2 correct refusals).

**Project hygiene**
- `.gitignore` excluding derived data, secrets, and virtual env.
- `README.md` documenting architecture, setup, and the reasoning behind
  each design decision (not just what the code does).
- Git repository initialized and pushed to
  [github.com/Geordie166/sop-copilot](https://github.com/Geordie166/sop-copilot).

## In progress

- **Commit authorship fix** — the first two commits were made before git
  identity was configured correctly and are attributed to a placeholder
  name. A `git filter-branch` + force-push to correct this is queued;
  blocked on being run manually due to a safety check on rewriting
  already-pushed history (see current chat for the exact commands).

## Not started / future work

- **GitHub Actions CI** — run `pytest` automatically on every push, so the
  test suite's status is visible on the repo itself rather than only when
  run locally.
- **LICENSE file** — standard for a public portfolio repo (e.g. MIT).
- **Grow the eval dataset** — 7 questions is enough to prove the pipeline
  works, not enough to be statistically meaningful. Worth expanding
  alongside the document corpus.
- **Optional: a simple demo UI** (e.g. Streamlit) — would make the project
  easier to show off in an interview/portfolio setting than a terminal
  CLI, at the cost of extra surface area to maintain.
- **Optional: persisted conversation history / multi-user support** — not
  needed to demonstrate the RAG pipeline itself; would matter for an
  actual deployment.

## Open design questions (for mentor review)

- **Hardcoded distance cutoff vs. model judgment for refusals.** Right now
  Claude decides whether search results are relevant enough to answer
  from, using the relevance distances in the tool result as one input to
  its own judgment (see `SYSTEM_PROMPT` in `src/agent.py`). A hardcoded
  numeric cutoff in code would be more predictable and auditable, but less
  flexible across differently-phrased questions. Deliberately left as
  model judgment for now — this is a real tradeoff worth discussing rather
  than one to resolve unilaterally, since it changes user-facing safety
  behavior for a compliance-adjacent assistant.
