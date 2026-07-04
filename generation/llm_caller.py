"""
generation/llm_caller.py

Handles all interaction with the Groq LLM API and parses the response
into a structured dict for the rest of the pipeline to consume.

Responsible for:
    - Sending the messages list to Groq
    - Measuring latency precisely
    - Extracting token usage from the API response
    - Parsing citations from the response text
    - Wrapping everything in try/except so API errors never crash the app

Why low temperature (0.1)?
    We want the model to extract and report, not creatively rephrase.
    High temperature increases linguistic variety but also increases
    the chance the model drifts from the exact citation format we
    specified, making parsing harder. 0.1 gives minimal creativity
    while avoiding the robotic repetitiveness of temperature=0.

Why parse citations from response text rather than using structured output?
    Groq's free tier supports standard chat completions reliably.
    Structured output (JSON mode) works but can cause the model to
    refuse to generate a natural-language answer alongside the JSON.
    Embedding citations inline as [Source: X, Page N] gives us both
    a readable answer and parseable citations in one response.
"""

import re
import time
from typing import List

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

# Regex to extract citations in the format: [Source: filename.pdf, Page 42]
# Tolerates minor formatting variations (extra spaces, missing page, etc.)
CITATION_PATTERN = re.compile(
    r'\[Source:\s*([^,\]]+),\s*Page\s*(\d+)\]',
    re.IGNORECASE,
)


def _parse_citations(text: str) -> List[dict]:
    """
    Extract all citations from the LLM response text.

    Parses inline citations formatted as [Source: FILENAME, Page N]
    and returns them as structured dicts for storage in the query log
    and display in the UI.

    Args:
        text: Raw response string from the LLM.

    Returns:
        List of dicts: [{filename: str, page: int}, ...]
        Deduplicated — the same source cited twice appears once.
    """
    seen = set()
    citations = []

    for match in CITATION_PATTERN.finditer(text):
        filename = match.group(1).strip()
        page = int(match.group(2))
        key = (filename, page)

        if key not in seen:
            seen.add(key)
            citations.append({"filename": filename, "page": page})

    return citations


def generate_response(messages: list) -> dict:
    """
    Send a messages list to Groq and return a structured response dict.

    The messages list comes from prompt_builder.build_prompt()['messages'].
    This function knows nothing about retrieval or prompt construction —
    it only handles the API call and response parsing.

    Token counts are extracted directly from the API response object.
    Groq returns these reliably and they're useful for cost estimation
    and monitoring (Phase 6 logs them to SQLite).

    Args:
        messages: List of {role, content} dicts (system + user messages).

    Returns:
        Dict with:
            response      (str)   : LLM answer text
            citations     (list)  : parsed citation dicts
            latency       (float) : wall-clock seconds for the API call
            input_tokens  (int)   : tokens in the prompt
            output_tokens (int)   : tokens in the completion
            error         (str|None): error message if the call failed

    Note:
        Never raises — on any exception, returns a safe error response
        so the pipeline degrades gracefully instead of crashing the UI.
    """
    client = Groq(api_key=GROQ_API_KEY)

    start = time.time()

    try:
        api_response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.1,   # low: factual extraction, not creative writing
        )

        latency = time.time() - start
        response_text = api_response.choices[0].message.content

        return {
            "response": response_text,
            "citations": _parse_citations(response_text),
            "latency": round(latency, 3),
            "input_tokens": api_response.usage.prompt_tokens,
            "output_tokens": api_response.usage.completion_tokens,
            "error": None,
        }

    except Exception as e:
        latency = time.time() - start
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[LLM_CALLER] API error: {error_msg}")

        return {
            "response": "I encountered an error generating a response. Please try again.",
            "citations": [],
            "latency": round(latency, 3),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": error_msg,
        }