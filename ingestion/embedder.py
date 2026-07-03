"""
ingestion/embedder.py

Converts text chunks into vector embeddings using sentence-transformers
and stores them in a FAISS index for fast similarity search at query time.

Two files are written to data/vector_store/:
    faiss_index.bin — the FAISS index (stores vectors only, not text)
    metadata.pkl    — chunk texts + metadata dicts (parallel to index)

Why two files? FAISS is a pure vector store. It knows vectors and distances,
nothing else. When we retrieve the top-k closest vectors to a query, FAISS
returns integer indices (0, 3, 7, ...). We use those indices to look up the
actual text and metadata from the pickle file. This is the standard pattern
for any FAISS-backed retrieval system.

Why IndexFlatIP with normalized vectors instead of IndexFlatL2?
IndexFlatL2 measures Euclidean distance (straight-line distance in 384D space).
IndexFlatIP measures inner product (dot product). When vectors are L2-normalized
to unit length first, inner product becomes mathematically identical to cosine
similarity. Cosine similarity is preferred for text because it measures the
*angle* between vectors (semantic direction) rather than absolute distance,
making it robust to text length differences. Phase 3 explains this in depth.
"""

import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR

INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.pkl"


def get_embedding_model() -> SentenceTransformer:
    """
    Load the sentence-transformer embedding model.

    First run: downloads ~90MB from HuggingFace Hub and caches locally.
    Subsequent runs: loads from cache in under a second.

    Returns:
        Loaded SentenceTransformer model instance.
    """
    print(f"[EMBEDDER] Loading model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_and_store(chunks: List[Document]) -> None:
    """
    Embed all chunks and persist the FAISS index + metadata to disk.

    Steps:
        1. Extract raw text strings from Document objects.
        2. Batch-encode them into 384-dim float32 vectors.
        3. L2-normalize each vector (required for cosine similarity via IP).
        4. Build a flat FAISS index and add all vectors.
        5. Write index to faiss_index.bin.
        6. Write text + metadata to metadata.pkl (parallel structure).

    Args:
        chunks: List of chunked Document objects to embed and store.

    Raises:
        ValueError: If chunks list is empty.
    """
    if not chunks:
        raise ValueError("Cannot embed empty chunk list. Load PDFs first.")

    model = get_embedding_model()
    texts = [chunk.page_content for chunk in chunks]

    print(f"[EMBEDDER] Generating embeddings for {len(texts)} chunks...")
    raw_embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=64,        # process 64 chunks at a time; adjust if RAM is tight
        convert_to_numpy=True,
    )

    # Cast to float32 — FAISS requires this; sentence-transformers may return float64
    embeddings = np.array(raw_embeddings).astype(np.float32)

    # Normalize to unit length in-place so dot product = cosine similarity
    faiss.normalize_L2(embeddings)

    # Build the index — IndexFlatIP = exact search using inner product
    index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    index.add(embeddings)

    # Persist the index
    faiss.write_index(index, str(INDEX_PATH))
    print(f"[EMBEDDER] FAISS index saved → {INDEX_PATH} ({index.ntotal} vectors)")

    # Persist metadata parallel to the index (same order, same indices)
    metadata = [
        {
            "text": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[EMBEDDER] Metadata saved → {METADATA_PATH}")


def load_index() -> Tuple[faiss.Index, List[dict]]:
    """
    Load a previously saved FAISS index and its paired metadata.

    Called by the retrieval layer (Phase 3), not by the ingestion pipeline.
    Kept here because the embedder owns the storage format — retrieval
    should not need to know whether we used pickle, JSON, or a database.

    Returns:
        Tuple of (faiss_index, metadata_list).
        metadata_list[i] corresponds to the vector at index i in faiss_index.

    Raises:
        FileNotFoundError: If ingestion hasn't been run yet.
    """
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Vector store not found. Run the ingestion pipeline first:\n"
            "  python main.py --ingest"
        )

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"[EMBEDDER] Loaded FAISS index: {index.ntotal} vectors")
    return index, metadata