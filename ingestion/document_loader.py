"""
ingestion/document_loader.py

Loads PDF files from the documents directory using PyMuPDF (fitz),
extracts text page-by-page, and returns LangChain Document objects
with metadata needed for access control and citations downstream.

Design decision — why PyMuPDF over LangChain's built-in PDF loader:
PyMuPDF gives us direct control over the extraction process (page numbers,
blank-page filtering, encoding). LangChain's community loaders are convenient
wrappers but add an extra abstraction layer we don't need here.
"""

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document

from config import DOCUMENTS_DIR


def infer_category(filename: str) -> str:
    """
    Infer document category from filename prefix convention.

    Convention: name files with their category as a prefix, e.g.
        finance_q3_report.pdf  → category: 'finance'
        hr_onboarding_guide.pdf → category: 'hr'
        general_faq.pdf        → category: 'general'

    This is intentionally simple — in a production system you would
    store category in a database or document registry instead.

    Args:
        filename: The PDF filename (just the name, not the full path).

    Returns:
        Category string: 'finance', 'hr', or 'general'.
    """
    name_lower = filename.lower()
    for category in ["finance", "hr"]:
        if name_lower.startswith(category):
            return category
    return "general"


def load_pdf(pdf_path: Path) -> List[Document]:
    """
    Load a single PDF and extract its text page-by-page.

    Each page becomes one Document object. Blank pages are skipped —
    they would produce empty embeddings that pollute the vector store
    with zero-signal noise.

    Args:
        pdf_path: Absolute Path to the PDF file.

    Returns:
        List of Document objects (one per non-blank page) with metadata:
            - source   : full file path string (for citations)
            - filename : just the file name   (for display)
            - page     : 1-indexed page number (human-readable)
            - category : inferred from filename (for access control)
    """
    documents = []
    category = infer_category(pdf_path.name)

    with fitz.open(str(pdf_path)) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text().strip()

            if not text:
                continue  # skip blank/image-only pages

            doc = Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "page": page_num + 1,   # 1-indexed for human readability
                    "category": category,
                },
            )
            documents.append(doc)

    return documents


def load_all_pdfs() -> List[Document]:
    """
    Load every PDF file found in the configured documents directory.

    Returns:
        Combined list of Documents from all PDFs. Empty list (not an
        error) if no PDFs exist yet — the pipeline should handle this
        gracefully rather than crashing on an empty documents folder.
    """
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in {DOCUMENTS_DIR}")
        print("[WARNING] Add PDFs to data/documents/ before running ingestion.")
        return []

    all_documents = []
    for pdf_path in pdf_files:
        print(f"[LOADER] Loading: {pdf_path.name}")
        docs = load_pdf(pdf_path)
        all_documents.extend(docs)
        print(f"[LOADER] Extracted {len(docs)} pages from {pdf_path.name}")

    print(f"[LOADER] Total pages loaded across all PDFs: {len(all_documents)}")
    return all_documents