# SOP Copilot

A small Retrieval-Augmented Generation (RAG) project that answers questions
about internal process documentation — standard operating procedures (SOPs)
and root cause analysis (RCA) reports — using the Claude API and a local
ChromaDB vector store.

Built as a learning project for a transition into AI Engineering. The goal
was to build a full, working RAG pipeline end to end rather than a toy demo:
real chunking tradeoffs, a real vector store, and a real Claude tool-use
loop with grounding/citation rules, not just a single prompt-and-response
script.

## Domain

The source documents simulate a health plan member services call center: 3
SOPs (HIPAA identity verification, call escalation, after-call work
documentation) and 2 RCAs (elevated handle time, repeat-call volume). All
content is fictional — no real member data, no real company.

## Architecture

```
data/raw/            Source SOP/RCA .txt documents (source of truth)
data/processed/       Generated ChromaDB store (derived, gitignored)
src/ingest.py         Load .txt files, chunk them into overlapping windows
src/embed.py          Embed chunks into ChromaDB, expose a query function
src/agent.py          Claude tool-use loop + interactive CLI / --demo mode
tests/test_ingest.py  Unit tests for the deterministic chunking logic
tests/test_embed.py   Unit tests for ChromaDB storage/retrieval (local only)
tests/test_agent.py   Unit tests for the tool-use loop (mocked Anthropic client)
```

Pipeline: **ingest** (load + chunk) → **embed** (vectorize + store in
ChromaDB) → **agent** (Claude decides when to search, retrieves relevant
chunks, answers with citations).

## Key design decisions

- **Word-based chunking, not character- or sentence-based.** Word count is
  a simple, easy-to-reason-about proxy for token count without needing a
  real tokenizer at this stage. Default is 500 words with 50-word overlap
  so ideas near a chunk boundary aren't lost entirely to one side.
- **Local embedding model (ChromaDB's bundled `all-MiniLM-L6-v2`), not an
  API-based one.** Claude's API doesn't expose a text-embeddings endpoint,
  and adding a second provider (e.g. OpenAI) just for embeddings would add
  cost and complexity this project doesn't need.
- **Deterministic chunk IDs + `upsert`, not auto-generated IDs + `add`.**
  Re-running `embed.py` after editing a source document updates that
  document's entries in place instead of accumulating duplicates.
- **Citation and "don't guess" rules live in the system prompt, not in
  code.** Claude reasons over the actual relevance distances returned by
  the search tool rather than a hardcoded numeric cutoff — simpler, but an
  open question for further discussion (see below).
- **`embed.py` is tested against a real (but temporary, isolated) local
  ChromaDB store, while `agent.py` is tested with a mocked Anthropic
  client.** Embedding/retrieval is entirely local (no API key or network
  call to Claude involved), so it's cheap to test for real. Claude's
  actual responses are not: `agent.py`'s tests mock `client.messages.create`
  to verify the loop-control logic (message history, stopping condition,
  the iteration cap) deterministically and for free. Whether Claude's
  *answers* are actually good (citing sources, refusing out-of-scope
  questions) is still checked the only way that's meaningful — by running
  `python src/agent.py --demo` against the real API.
- **Conversation history is scoped to a CLI session, not persisted to
  disk.** `run_interactive()` keeps a single `messages` list alive for the
  life of the REPL so follow-up questions have context, but nothing is
  saved between runs — there's no multi-user or resumed-session
  requirement yet to justify that complexity.

## Setup

```
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Create a `.env` file with:

```
ANTHROPIC_API_KEY=your-key-here
```

## Running it

```
python src/ingest.py     # preview document loading + chunking
python src/embed.py      # embed all documents into ChromaDB, run a test query
python src/agent.py      # interactive chat session (multi-turn, type 'exit' to quit)
python src/agent.py --demo  # non-interactive: two canned questions against the real API
pytest tests/            # run the full unit test suite (15 tests, no API key required)
```

## Open questions for further review

- Should the "no relevant document found" decision move from the system
  prompt (model judgment) into code (a hard distance-based cutoff)? Model
  judgment is more flexible across phrasings; a hard cutoff is more
  predictable and auditable — worth a real tradeoff discussion. Left as
  model judgment for now; deliberately not resolved unilaterally since it
  changes user-facing safety behavior.
- No persistence of conversation history across CLI sessions, and no
  multi-user support — both would matter for a real deployment but aren't
  needed to demonstrate the RAG pipeline itself.
- No evaluation harness (e.g. a labeled set of Q&A pairs to measure
  retrieval/answer quality over time) — worth adding if this project grows
  beyond a handful of example documents.
