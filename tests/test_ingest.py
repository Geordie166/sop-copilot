"""
test_ingest.py — unit tests for the chunking/loading logic in src/ingest.py.

WHY these tests target ingest.py specifically (and not embed.py or
agent.py): chunk_text and load_documents are pure, deterministic functions
with no network calls — no Claude API, no ChromaDB, no embedding model
download. That makes them fast, free, and reliable to test automatically.
embed.py and agent.py both depend on external services (a local embedding
model download and live Claude API calls respectively); testing those
meaningfully would mean mocking the Anthropic client and ChromaDB, which
is a reasonable next step but a bigger investment than this first test
pass — noted as a follow-up rather than done here.
"""

from ingest import chunk_text, load_documents


def _words(n: int) -> str:
    """Build a space-joined string of n distinct, easy-to-index words."""
    return " ".join(f"word{i}" for i in range(n))


def test_chunk_text_returns_single_chunk_when_text_shorter_than_chunk_size():
    text = _words(50)

    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # WHY assert exactly one chunk: this is the short-circuit path in
    # chunk_text — most of this project's real SOP/RCA documents are under
    # the default chunk_size, so this is actually the common case, not an
    # edge case.
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_splits_long_text_into_multiple_overlapping_chunks():
    text = _words(25)

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    # With 25 words, chunk_size=10, overlap=2 (step=8), the sliding window
    # produces windows starting at 0, 8, and 16 -> 3 chunks.
    assert len(chunks) == 3
    assert chunks[0] == "word0 word1 word2 word3 word4 word5 word6 word7 word8 word9"
    assert chunks[1].startswith("word8 word9")
    assert chunks[2].startswith("word16")


def test_chunk_text_overlap_repeats_boundary_words_between_chunks():
    text = _words(25)

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    # WHY this is the test that actually proves overlap works: the last
    # `overlap` words of one chunk must equal the first `overlap` words of
    # the next chunk. This is the behavior the overlap parameter exists to
    # guarantee — that no idea is fully lost by being split across a chunk
    # boundary with zero shared context.
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-2:] == second_chunk_words[:2]


def test_chunk_text_covers_every_word_with_no_gaps():
    text = _words(37)

    chunks = chunk_text(text, chunk_size=10, overlap=3)

    # WHY check the last chunk reaches the final word rather than counting
    # total words across chunks: overlap means words are intentionally
    # repeated, so a total word count wouldn't equal len(text.split()).
    # What actually matters is that the sliding window doesn't stop short
    # and silently drop trailing content.
    assert chunks[-1].split()[-1] == "word36"


def test_load_documents_reads_all_txt_files_in_directory(tmp_path):
    (tmp_path / "doc_a.txt").write_text("Alpha content", encoding="utf-8")
    (tmp_path / "doc_b.txt").write_text("Beta content", encoding="utf-8")
    # WHY include a non-.txt file: load_documents globs for "*.txt"
    # specifically, so this confirms it doesn't accidentally pick up
    # unrelated files that might live in data/raw (e.g. a README or .DS_Store).
    (tmp_path / "notes.md").write_text("# not a source doc", encoding="utf-8")

    documents = load_documents(raw_dir=tmp_path)

    assert set(documents.keys()) == {"doc_a.txt", "doc_b.txt"}
    assert documents["doc_a.txt"] == "Alpha content"
    assert documents["doc_b.txt"] == "Beta content"
