"""
retrieval/retriever.py

Orchestration layer that combines vector search with access control.

This module answers: "given this user and this query, what chunks
are they allowed to see that are also semantically relevant?"

It deliberately knows nothing about how embeddings work (that's
vector_store.py's job) and nothing about how access rules are stored
(that's access_control.py's job in Phase 4). It only orchestrates.
"""

import json
from typing import List

from config import TOP_K_RESULTS, USERS_FILE
from retrieval.vector_store import search


def get_allowed_categories(username: str) -> List[str]:
    """
    Load a user's allowed document categories from users.json.

    Kept here as a lightweight convenience until Phase 4 builds the
    full access_control module. Phase 4 will replace this with a call
    to access_control.get_allowed_categories() instead.

    Args:
        username: The username to look up.

    Returns:
        List of category strings the user can access.
        Returns ['general'] for unknown users as a safe fallback.
    """
    with open(USERS_FILE, "r") as f:
        users = json.load(f)

    if username not in users:
        print(f"[RETRIEVER] Unknown user '{username}', defaulting to general access.")
        return ["general"]

    return users[username]["allowed_categories"]


def retrieve(query: str, username: str, top_k: int = TOP_K_RESULTS) -> List[dict]:
    """
    Retrieve the top-k most relevant chunks a given user is allowed to see.

    Pipeline:
        1. Fetch allowed categories for this user from users.json.
        2. Call vector_store.search() with an inflated top_k (over-retrieve).
        3. Filter results to only chunks whose category is allowed.
        4. Return the top_k of the filtered set, still sorted by relevance.

    Why over-retrieve in step 2?
    FAISS searches the full index regardless of category. If we ask for
    exactly 5 results and all 5 happen to be from a restricted category,
    the filter would return nothing. Fetching TOP_K_RESULTS * 4 candidates
    first gives the filter enough material to always return a full result set.

    Args:
        query:    Natural language query from the user.
        username: Logged-in username for access control.
        top_k:    Number of final results to return after filtering.

    Returns:
        List of up to top_k dicts, each with 'text', 'metadata', 'score'.
        Empty list if no relevant accessible chunks are found.
    """
    allowed_categories = get_allowed_categories(username)
    print(f"[RETRIEVER] User '{username}' | allowed: {allowed_categories}")

    # Over-retrieve so the filter has enough candidates
    candidates = search(query, top_k=top_k * 4)

    # Filter to only chunks the user is allowed to access
    filtered = [
        chunk for chunk in candidates
        if chunk["metadata"].get("category", "general") in allowed_categories
    ]

    # Candidates are already sorted by score descending from FAISS,
    # so slicing gives the top-k most relevant accessible chunks.
    results = filtered[:top_k]

    print(f"[RETRIEVER] {len(candidates)} candidates → {len(filtered)} after filter → {len(results)} returned")

    return results