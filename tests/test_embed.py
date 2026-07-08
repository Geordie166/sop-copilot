"""
test_embed.py — unit tests for the ChromaDB storage/retrieval layer in
src/embed.py.

WHY these tests don't need an Anthropic API key: embedding and storage are
entirely local — ChromaDB's bundled embedding model runs on-device via
onnxruntime, no network call to Claude is involved. That makes this layer
testable without hitting a paid API, unlike agent.py (see test_agent.py,
which mocks the Anthropic client instead).

WHY tests use a temp ChromaDB path instead of the real data/processed/
store: reusing the real store would make tests depend on (and pollute)
whatever's actually been ingested during manual runs. Each test gets an
isolated, throwaway collection via monkeypatching CHROMA_DB_PATH.
"""

import embed


def _fresh_collection(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "CHROMA_DB_PATH", str(tmp_path / "chroma_db"))
    return embed.get_chroma_collection()


def test_get_chroma_collection_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "CHROMA_DB_PATH", str(tmp_path / "chroma_db"))

    collection_a = embed.get_chroma_collection()
    collection_a.upsert(ids=["x"], documents=["hello world"], metadatas=[{"source": "a.txt", "chunk_index": 0}])

    # WHY call get_chroma_collection() a second time rather than reusing
    # collection_a: get_or_create_collection must be safe to call again on
    # a collection that already has data in it (this is exactly what
    # embed.py's __main__ block does every time it's rerun) — it should
    # reattach to the same underlying data, not error or reset it.
    collection_b = embed.get_chroma_collection()

    assert collection_b.count() == 1


def test_query_sop_documents_ranks_the_semantically_closest_chunk_first(tmp_path, monkeypatch):
    collection = _fresh_collection(tmp_path, monkeypatch)

    collection.upsert(
        ids=["doc1::chunk0", "doc2::chunk0", "doc3::chunk0"],
        documents=[
            "Steps to verify a caller's identity before discussing protected health information.",
            "Procedure for changing the battery on an electric forklift.",
            "Instructions for purging resin from an injection molding machine barrel.",
        ],
        metadatas=[
            {"source": "doc1.txt", "chunk_index": 0},
            {"source": "doc2.txt", "chunk_index": 0},
            {"source": "doc3.txt", "chunk_index": 0},
        ],
    )

    results = embed.query_sop_documents(
        collection, "how do I confirm who I'm speaking with before sharing member data", n_results=2
    )

    assert len(results) == 2
    # WHY assert only on the top result's source rather than exact
    # distances: exact embedding distances depend on model internals that
    # could shift with a model/library version bump; what actually matters
    # for this function's contract is that the *most relevant* document
    # comes back first.
    assert results[0]["source"] == "doc1.txt"
    assert set(results[0].keys()) == {"text", "source", "chunk_index", "distance"}


def test_query_sop_documents_respects_n_results(tmp_path, monkeypatch):
    collection = _fresh_collection(tmp_path, monkeypatch)

    collection.upsert(
        ids=[f"doc::chunk{i}" for i in range(5)],
        documents=[f"sample chunk number {i} about a warehouse topic" for i in range(5)],
        metadatas=[{"source": "doc.txt", "chunk_index": i} for i in range(5)],
    )

    results = embed.query_sop_documents(collection, "warehouse topic", n_results=2)

    assert len(results) == 2
