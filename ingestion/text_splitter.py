"""
ingestion/text_splitter.py

Splits LangChain Document objects (one per PDF page) into smaller,
overlapping chunks using RecursiveCharacterTextSplitter.

Why "Recursive"? The splitter tries a list of separators in priority order:
    1. Paragraph breaks (\n\n)  — best: keeps paragraphs together
    2. Line breaks (\n)          — good: keeps sentences together
    3. Period+space (". ")       — okay: splits at sentence boundaries
    4. Spaces (" ")              — fallback: splits at word boundaries
    5. Characters ("")           — last resort: hard split mid-word

It recurses down this list until the chunk fits within chunk_size.
This makes chunks far more semantically coherent than a naive
fixed-character split that ignores text structure entirely.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller overlapping chunks.

    Chunk size and overlap are configured centrally in config.py
    (CHUNK_SIZE=500, CHUNK_OVERLAP=50) so they can be tuned without
    touching pipeline code.

    Why chunk_overlap > 0?
    Imagine a sentence that straddles the boundary between two chunks.
    Without overlap, the first chunk gets its beginning and the second
    gets its ending — neither has the complete thought. A 50-char overlap
    ensures each chunk shares a small "seam" with its neighbors so
    cross-boundary context isn't lost.

    All original metadata (filename, page, category, source) is
    automatically carried through to every child chunk by LangChain's
    splitter — this is critical for citations later.

    Args:
        documents: List of Document objects (typically one per PDF page).

    Returns:
        List of chunked Documents. Each chunk inherits the parent page's
        metadata plus a 'chunk_index' field for debugging.
    """
    if not documents:
        print("[SPLITTER] No documents to split.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Attach a global chunk index for traceability during debugging
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"[SPLITTER] {len(documents)} pages → {len(chunks)} chunks")
    print(f"[SPLITTER] Avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

    return chunks