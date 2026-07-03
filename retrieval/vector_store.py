"""
retrieval/vector_store.py

Pure vector search layer — knows about embeddings and FAISS, nothing else.
Has no concept of users, categories, or access control; those concerns
belong to retriever.py above this layer.

Design principle: keep this module swappable. If you later replace FAISS
with Pinecone or Qdrant, only this file changes. The retriever and everything
above it stays identical.
"""

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, TOP_K_RESULTS
from ingestion.embedder import load_index

# Module-level cache — load the index and model once per process,
# not once per query. Loading FAISS from disk + initializing the
# sentence-transformer model both take ~1s; doing that per query
# would make the system feel broken.
_index: faiss.Index = None
_metadata: List[dict] = None
_model: SentenceTransformer = None


def _get_resources():
    """
    Lazily load and cache the FAISS index, metadata, and embedding model.

    Uses module-level globals so the expensive initialization only happens
    on the first call. Every subsequent call returns the cached objects
    instantly. This pattern is called lazy initialization.

    Returns:
        Tuple of (faiss_index, metadata_list, embedding_model)
    """
    global _index, _metadata, _model

    if _index is None:
        _index, _metadata = load_index()

    if _model is None:
        print(f"[VECTOR_STORE] Loading embedding model: {EMBEDDING_MODEL_NAME}")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _index, _metadata, _model


def search(query: str, top_k: int = None) -> List[dict]:
    """
    Embed a query and return the top-k most similar chunks from the index.

    We over-retrieve by default (fetching more than TOP_K_RESULTS) to give
    the access-control filter in retriever.py enough candidates to work with.
    If the filter discards some results, we still return a meaningful set.

    Similarity score interpretation (cosine, range -1 to 1):
        > 0.7  : highly relevant, strong semantic match
        0.4-0.7: moderately relevant
        < 0.4  : weak match, the answer likely is not in these chunks

    Args:
        query:  The user's natural language question.
        top_k:  Number of candidates to retrieve. Defaults to TOP_K_RESULTS * 4
                so the category filter has room to work.

    Returns:
        List of dicts, sorted by score descending, each containing:
            - text     : the raw chunk text
            - metadata : dict with filename, page, category, chunk_index
            - score    : cosine similarity score (float, 0-1 after normalization)
    """
    if top_k is None:
        top_k = TOP_K_RESULTS * 4  # over-retrieve; filter happens in retriever.py

    index, metadata, model = _get_resources()

    # Embed the query using the same model used during ingestion.
    # Using a different model would produce incompatible vector spaces.
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)

    # Normalize so the inner product equals cosine similarity
    # (matches how we stored vectors in embedder.py)
    faiss.normalize_L2(query_embedding)

    # FAISS search: returns (scores, indices) arrays of shape (1, top_k)
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when fewer than top_k vectors exist in the index
            continue

        entry = metadata[idx]
        results.append({
            "text": entry["text"],
            "metadata": entry["metadata"],
            "score": float(score),
        })

    return results