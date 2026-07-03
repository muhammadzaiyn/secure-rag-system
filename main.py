"""
main.py

Entry point for the Secure RAG System.
Runs the full ingestion pipeline: load PDFs → chunk → embed → save index.

Later phases will add retrieval, guardrails, and generation here.
For now this is purely the ingestion runner used to test Phase 2.

Usage:
    python main.py
"""

from config import validate_config
from ingestion.document_loader import load_all_pdfs
from ingestion.text_splitter import split_documents
from ingestion.embedder import embed_and_store


def run_ingestion() -> None:
    """
    Execute the full document ingestion pipeline.

    Load → Chunk → Embed → Store.
    Safe to re-run: overwrites the existing index each time,
    which is the correct behaviour when documents are added or changed.
    """
    print("=" * 50)
    print("  Secure RAG System — Ingestion Pipeline")
    print("=" * 50)

    # Step 1: Load all PDFs from data/documents/
    documents = load_all_pdfs()
    if not documents:
        print("\n[STOP] No documents loaded. Add PDFs to data/documents/ and retry.")
        return

    # Step 2: Chunk pages into smaller overlapping pieces
    chunks = split_documents(documents)

    # Step 3: Embed chunks and save FAISS index to data/vector_store/
    embed_and_store(chunks)

    print("\n[DONE] Ingestion complete. Vector store is ready for retrieval.")
    print(f"       Index contains {len(chunks)} chunks from {len(documents)} pages.")


if __name__ == "__main__":
    validate_config()
    run_ingestion()