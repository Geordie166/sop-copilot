"""
ingest.py — Document loading and chunking for SOP Copilot's RAG pipeline.

WHY THIS MODULE EXISTS:
Before we can embed documents into a vector store, we need to (1) load raw
text off disk and (2) break it into "chunks" small enough to embed
meaningfully and retrieve precisely. A whole SOP document is too coarse a
unit to retrieve — if a user asks "what PPE is required for battery
changeout," we want to return the relevant paragraph, not the entire
12-step procedure. Chunking is what makes retrieval precise instead of
returning whole documents every time.
"""

import os
from pathlib import Path

# WHY a constant path instead of a hardcoded string inline: keeping the raw
# data directory as a single named constant means every function that needs
# it stays in sync if the folder ever moves, and it documents the expected
# project layout at a glance.
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_documents(raw_dir: Path = RAW_DATA_DIR) -> dict[str, str]:
    """
    Load every .txt file in raw_dir into memory.

    Returns a dict of {filename: full_text}. WHY a dict keyed by filename
    rather than a flat list of strings: we need to preserve which document
    each chunk came from later (for citation/source metadata), and keying
    by filename now is the simplest way to keep that link intact through
    the rest of the pipeline.
    """
    documents = {}

    for file_path in sorted(raw_dir.glob("*.txt")):
        # WHY encoding="utf-8" explicitly: relying on the OS default
        # encoding (which can be a legacy codepage on Windows) is a classic
        # source of silent mojibake when a file contains anything outside
        # plain ASCII. Being explicit here makes behavior consistent across
        # machines.
        text = file_path.read_text(encoding="utf-8")
        documents[file_path.name] = text

    return documents


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping, word-based chunks.

    WHY word-based (not character-based or sentence-based) chunking:
    word count is a reasonable, easy-to-reason-about proxy for token count
    (roughly 0.75 tokens per word for English text), without needing a real
    tokenizer at this stage. It's simple to implement correctly and good
    enough for a first RAG pass — a production system would likely switch
    to a tokenizer-aware or semantic (paragraph/section-based) chunker, but
    that's a later optimization, not a prerequisite for a working pipeline.

    WHY overlap matters: if we chunked with hard, non-overlapping
    boundaries, a sentence or procedural step that happens to fall right at
    a chunk boundary gets split across two chunks — and neither half
    contains enough context on its own for a similarity search to match it
    well, or for Claude to reason over it coherently. Overlap (default 50
    words) means each chunk repeats the tail end of the previous chunk, so
    boundary-spanning ideas stay readable in at least one chunk.
    """
    words = text.split()

    if len(words) <= chunk_size:
        # WHY short-circuit here: most of our example SOPs/RCAs are under
        # 400 words, i.e. smaller than the default chunk_size. Running them
        # through the sliding-window loop below would still produce exactly
        # one chunk, but returning early makes that common case explicit
        # and avoids any off-by-one edge cases in the loop for tiny inputs.
        return [text.strip()]

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

        # WHY advance by (chunk_size - overlap) instead of chunk_size:
        # advancing by the full chunk_size would make overlap pointless,
        # since the next window wouldn't actually revisit any prior words.
        # Stepping back by `overlap` words each time is what creates the
        # repeated boundary context described above.
        start += chunk_size - overlap

    return chunks


def chunk_documents(
    documents: dict[str, str], chunk_size: int = 500, overlap: int = 50
) -> dict[str, list[str]]:
    """
    Apply chunk_text to every loaded document.

    Returns {filename: [chunk_1, chunk_2, ...]}. Kept as a thin wrapper
    rather than inlining this loop at the call site so ingest.py has one
    obvious entry point for "give me chunks for all documents," which
    embed.py will import and call directly.
    """
    return {
        filename: chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for filename, text in documents.items()
    }


if __name__ == "__main__":
    # WHY a __main__ block instead of a separate script: this lets ingest.py
    # be run directly for a quick sanity check ("did my documents load and
    # chunk the way I expect?") while still being importable as a module
    # from embed.py without executing this preview code on import.
    documents = load_documents()
    chunked = chunk_documents(documents)

    print(f"Loaded {len(documents)} document(s) from {RAW_DATA_DIR}\n")

    for filename, chunks in chunked.items():
        print(f"--- {filename} ---")
        print(f"Chunk count: {len(chunks)}")
        preview = chunks[0][:200].replace("\n", " ")
        print(f"First chunk preview: {preview}...")
        print()
