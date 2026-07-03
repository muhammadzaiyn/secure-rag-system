"""
guardrails/toxicity.py

Detects toxic, abusive, or harmful content in user queries using a
locally-running HuggingFace classifier. No API call required — the
model runs entirely on your machine.

Model: martin-ha/toxic-comment-model
    - Fine-tuned BERT on the Wikipedia Toxic Comments dataset
    - Labels: 'toxic' or 'non_toxic'
    - Lightweight (~250MB), fast on CPU (~200-400ms per query)
    - Trade-off: trained on comment data, so may underperform on
      domain-specific toxicity (e.g. financial fraud language).
      Acceptable for a portfolio project; production would use a
      larger, domain-adapted model.

Why run locally instead of using an API?
    1. No latency added by a network round-trip for every query
    2. No cost per query
    3. No data leaves your machine (important for enterprise deployments
       where queries may contain confidential document content)
    4. No external dependency that could go down
"""

import time
from typing import Optional

from transformers import pipeline

from config import TOXICITY_MODEL_NAME, TOXICITY_THRESHOLD

# Module-level cache — same pattern as vector_store.py.
# The pipeline downloads and loads the model on first call (~2-3s),
# then subsequent calls are fast (~200-400ms).
_toxicity_pipeline = None


def _get_pipeline():
    """
    Lazily initialise and cache the toxicity classification pipeline.

    Returns:
        Loaded HuggingFace pipeline instance.
    """
    global _toxicity_pipeline
    if _toxicity_pipeline is None:
        print(f"[TOXICITY] Loading model: {TOXICITY_MODEL_NAME}")
        _toxicity_pipeline = pipeline(
            "text-classification",
            model=TOXICITY_MODEL_NAME,
        )
    return _toxicity_pipeline


def detect_toxicity(query: str) -> dict:
    """
    Classify a query as toxic or non-toxic.

    The model returns a label and a confidence score. We normalise this
    into a single 'toxicity score' between 0.0 and 1.0 regardless of
    which label came back, so downstream code never has to know the
    model's internal label names.

    Truncation: the model has a 512-token limit. Very long queries are
    silently truncated by the pipeline — this is acceptable since toxic
    content is almost always detectable in the first few sentences.

    Args:
        query: Raw user query string.

    Returns:
        Dict with keys:
            is_toxic    (bool)  — True if toxicity score exceeds threshold
            score       (float) — Toxicity probability (0.0-1.0)
            latency_ms  (float) — Detection time in milliseconds
    """
    start = time.time()

    classifier = _get_pipeline()

    # Truncate to 512 tokens to stay within model limits
    result = classifier(query[:512])[0]

    label = result["label"]      # "toxic" or "non_toxic"
    confidence = result["score"] # model's confidence in that label

    # Normalise: if label is "toxic", score = confidence.
    # If label is "non_toxic", toxicity score = 1 - confidence.
    toxicity_score = confidence if label == "toxic" else 1.0 - confidence

    latency_ms = (time.time() - start) * 1000

    return {
        "is_toxic": toxicity_score > TOXICITY_THRESHOLD,
        "score": round(toxicity_score, 4),
        "latency_ms": round(latency_ms, 1),
    }