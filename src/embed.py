"""
embed.py — Embedding storage and retrieval for SOP Copilot's RAG pipeline.

WHY THIS MODULE EXISTS:
ingest.py turns raw documents into text chunks. This module takes those
chunks, converts each one into a vector embedding (a numeric representation
of its meaning), and stores those vectors in ChromaDB so we can later find
the chunks most semantically similar to a user's question — even if the
question doesn't share exact keywords with the source text. That semantic
lookup is the "retrieval" half of Retrieval-Augmented Generation.
"""

import chromadb

from ingest import load_documents, chunk_documents

# WHY store the vector DB inside data/processed rather than data/raw:
# data/raw is meant to hold source-of-truth documents a human wrote or
# curated. The ChromaDB files are a *derived* artifact, entirely
# regenerable by re-running this script against data/raw. Keeping derived
# data out of data/raw (and out of version control, see .gitignore) keeps
# it clear which directory is authoritative.
CHROMA_DB_PATH = "data/processed/chroma_db"

# WHY a single fixed collection name as a constant: a "collection" in
# ChromaDB is roughly analogous to a table. Hardcoding the name here (once)
# means every function that touches the collection refers to the same
# constant instead of a repeated string literal that could drift out of
# sync if renamed later.
COLLECTION_NAME = "sop_documents"


def get_chroma_collection():
    """
    Initialize a persistent ChromaDB client and return the sop_documents
    collection, creating it if it doesn't exist yet.

    WHY PersistentClient (not the default in-memory client): an in-memory
    client would lose every embedding as soon as the Python process exits,
    which defeats the point of "ingest once, query many times later."
    PersistentClient writes to disk at CHROMA_DB_PATH so embeddings survive
    across separate runs of embed.py (and, eventually, the agent).

    WHY get_or_create_collection instead of create_collection: re-running
    this script (e.g. after adding a new document to data/raw) would
    otherwise raise an error on the second run because the collection
    already exists. get_or_create makes the script idempotent/rerunnable,
    which matters a lot during development.

    Note: we don't pass an explicit embedding_function here, so Chroma uses
    its bundled default (all-MiniLM-L6-v2, run locally via onnxruntime).
    WHY that default instead of calling out to an embeddings API: Claude's
    API doesn't currently expose a text-embeddings endpoint, and pulling in
    a second provider (e.g. OpenAI) just for embeddings would add cost and
    complexity this learning project doesn't need. The bundled model runs
    fully locally, is free, and is more than adequate for a small
    proof-of-concept corpus like this one.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def embed_and_store(collection) -> int:
    """
    Load, chunk, and embed every document in data/raw, storing results in
    the given ChromaDB collection.

    Returns the number of chunks stored.

    WHY we recompute IDs as "{filename}::chunk{index}" rather than letting
    Chroma auto-generate IDs: deterministic, content-derived IDs mean that
    re-running this script after editing a document updates that
    document's existing entries (Chroma upserts on a repeated ID) instead
    of silently accumulating duplicate chunks every time we re-ingest.
    """
    documents = load_documents()
    chunked = chunk_documents(documents)

    ids = []
    texts = []
    metadatas = []

    for filename, chunks in chunked.items():
        for i, chunk in enumerate(chunks):
            ids.append(f"{filename}::chunk{i}")
            texts.append(chunk)
            # WHY store source filename and chunk index as metadata rather
            # than only the embedding: metadata is what lets us tell the
            # user (or Claude, later) *which document* an answer came from,
            # which is essential for an SOP assistant — citing the source
            # procedure is often as important as the answer itself.
            metadatas.append({"source": filename, "chunk_index": i})

    # WHY collection.upsert instead of collection.add: upsert overwrites an
    # existing entry with the same ID instead of raising a duplicate-ID
    # error, which — combined with the deterministic IDs above — makes this
    # function safe to re-run any time documents change.
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

    return len(ids)


def query_sop_documents(collection, question: str, n_results: int = 3) -> list[dict]:
    """
    Run a natural-language query against the collection and return the
    top n_results most relevant chunks.

    WHY n_results defaults to 3: for a RAG system, returning too few
    chunks risks missing the answer if it's split across chunks or phrased
    differently than expected; returning too many dilutes the context
    Claude would eventually receive with less relevant material and costs
    more tokens. 3 is a common, reasonable starting point that we can
    tune later based on real answer quality.

    Returns a list of dicts (not the raw Chroma response) so callers don't
    need to know Chroma's nested-list response shape — this function is
    the boundary that translates "Chroma's API" into "a plain list of
    results," which will make it easier to swap retrieval backends later
    if needed without touching calling code.
    """
    results = collection.query(query_texts=[question], n_results=n_results)

    formatted = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        formatted.append(
            {
                "text": text,
                "source": metadata["source"],
                "chunk_index": metadata["chunk_index"],
                # WHY expose distance instead of hiding it: a lower distance
                # means a closer semantic match. Surfacing it lets us
                # sanity-check retrieval quality now (e.g. "is the top
                # result actually a good match, or just the least-bad
                # option?") rather than treating retrieval as a black box.
                "distance": distance,
            }
        )

    return formatted


if __name__ == "__main__":
    collection = get_chroma_collection()
    chunk_count = embed_and_store(collection)
    print(f"Embedded and stored {chunk_count} chunk(s) in '{COLLECTION_NAME}'\n")

    # WHY this query instead of the manufacturing-themed one originally
    # planned: the source corpus was switched to healthcare call center
    # documents partway through this project, so a conveyor-lockout query
    # would have no relevant match in the collection at all and wouldn't
    # meaningfully test retrieval. This question maps directly to
    # sop_hipaa_identity_verification.txt instead.
    test_question = "what are the steps to verify a caller's identity before discussing member information"
    print(f"Test query: {test_question!r}\n")

    results = query_sop_documents(collection, test_question, n_results=3)

    for rank, result in enumerate(results, start=1):
        print(f"--- Result {rank} (distance={result['distance']:.4f}) ---")
        print(f"Source: {result['source']} (chunk {result['chunk_index']})")
        preview = result["text"][:300].replace("\n", " ")
        print(f"Text: {preview}...")
        print()
