"""
generation/prompt_builder.py

Constructs the full prompt sent to the LLM for grounded response generation.

Responsible for:
    - Formatting retrieved chunks with their source metadata
    - Writing the system prompt that enforces grounding behavior
    - Assembling the final user message

This module has zero side effects — no API calls, no file I/O.
That makes it fully testable in isolation and easy to iterate on
prompt design without burning API tokens.

Why prompt design matters more than model choice in RAG:
    The retrieved chunks already contain the answer. The model's job
    is extraction and formatting, not reasoning from memory. A precise
    system prompt that enforces grounding will outperform a loose prompt
    with a stronger model — because the stronger model will hallucinate
    more confidently when given room to do so.
"""

from typing import List


# ---------------------------------------------------------------------------
# System prompt — the most important engineering artifact in this phase
# ---------------------------------------------------------------------------
# Every instruction here is deliberate. Each one closes a specific failure
# mode observed in production RAG systems:
#
#   "ONLY use information..."  → prevents hallucination
#   "I don't know..."          → prevents confident wrong answers
#   "never invent..."          → explicit prohibition on confabulation
#   Exact citation format      → makes citations machine-parseable
#   "Do not pad..."            → prevents verbose non-answers

SYSTEM_PROMPT = """You are a secure, grounded document assistant. \
Your only job is to answer questions using the document excerpts provided below.

You must follow these rules without exception:

1. ONLY use information explicitly present in the provided document excerpts.
2. If the answer cannot be found in the excerpts, respond with exactly:
   "I don't know based on the provided documents."
3. Never invent facts, statistics, names, dates, or figures not in the excerpts.
4. Cite every piece of information using this exact format:
   [Source: FILENAME, Page PAGE_NUMBER]
5. If multiple excerpts support your answer, cite all of them.
6. Be concise and direct. Do not pad your answer with filler phrases.
7. If excerpts are partially relevant but incomplete, say what you can confirm
   and explicitly state what is missing from the provided context.
"""


def _format_chunks(chunks: List[dict]) -> str:
    """
    Format retrieved chunks into a clearly delimited context block.

    Each chunk is prefixed with its source metadata so the model can
    produce accurate citations. The delimiter lines (---) signal clear
    boundaries between chunks to reduce cross-chunk confusion.

    Args:
        chunks: List of chunk dicts from retriever.retrieve(), each with
                'text' and 'metadata' keys.

    Returns:
        Formatted string block ready for insertion into the prompt.
    """
    if not chunks:
        return "No relevant document excerpts were found."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        filename = meta.get("filename", "unknown")
        page = meta.get("page", "?")
        score = chunk.get("score", 0.0)

        lines.append(f"[Excerpt {i} — {filename}, Page {page} | relevance: {score:.3f}]")
        lines.append(chunk["text"].strip())
        lines.append("")  # blank line between chunks

    return "\n".join(lines)


def build_prompt(query: str, chunks: List[dict], username: str) -> dict:
    """
    Assemble the complete prompt for the LLM as a messages list.

    Returns a messages list (system + user) rather than a single string
    because all modern LLM APIs (Groq, Anthropic, OpenAI) use the
    messages format. This keeps llm_caller.py clean and provider-agnostic.

    The user message deliberately separates the context block from the
    question with clear headers — this structural separation consistently
    improves citation accuracy in practice.

    Args:
        query:    The user's original question.
        chunks:   Retrieved document chunks from retriever.retrieve().
        username: The querying user (included for audit purposes in
                  the prompt — lets the model know the access context).

    Returns:
        Dict with:
            messages      : list of {role, content} dicts for the LLM API
            context_used  : formatted context string (for logging/debugging)
            chunks_count  : number of chunks included
    """
    context_block = _format_chunks(chunks)

    user_message = f"""DOCUMENT EXCERPTS:
---
{context_block}
---

USER QUESTION: {query}

Answer the question using only the excerpts above. \
Cite sources using [Source: FILENAME, Page PAGE_NUMBER] format."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    return {
        "messages": messages,
        "context_used": context_block,
        "chunks_count": len(chunks),
    }