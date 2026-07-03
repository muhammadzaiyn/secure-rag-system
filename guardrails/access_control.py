"""
guardrails/access_control.py

Authoritative access control layer for the Secure RAG System.

Replaces the lightweight get_allowed_categories() stub in retriever.py
(which was a Phase 3 placeholder). Everything that needs to know about
user permissions should import from this module, not from retriever.py.

Design note — why not a database?
    For a portfolio project, users.json is the right call: zero
    infrastructure, version-controllable, human-readable. In production
    you would store this in a proper identity provider (Auth0, Okta,
    or your own RBAC database) and call an authenticated API instead.
    The interface here (check_access / get_allowed_categories) is
    intentionally identical to what a production implementation would
    expose — only the backend storage would change.
"""

import json
from functools import lru_cache
from typing import List

from config import USERS_FILE


@lru_cache(maxsize=1)
def _load_users() -> dict:
    """
    Load and cache the users configuration from users.json.

    lru_cache(maxsize=1) caches the result of the first call.
    Subsequent calls return the cached dict instantly without re-reading
    the file. Cache is process-scoped — cleared on restart.

    This is appropriate here because user permissions don't change
    at runtime in this system. If they could change, you would use
    a time-based cache or invalidate on write instead.

    Returns:
        Dict mapping username -> {role, allowed_categories}.
    """
    with open(USERS_FILE, "r") as f:
        return json.load(f)


def get_allowed_categories(username: str) -> List[str]:
    """
    Return the list of document categories a user is permitted to access.

    Args:
        username: The username to look up.

    Returns:
        List of category strings. Returns ['general'] for unknown users
        as a safe, minimal-privilege fallback rather than raising an error —
        unknown users get read access to public/general documents only.
    """
    users = _load_users()

    if username not in users:
        print(f"[ACCESS_CONTROL] Unknown user '{username}' — defaulting to general only.")
        return ["general"]

    return users[username]["allowed_categories"]


def check_access(username: str, document_category: str) -> bool:
    """
    Check whether a specific user can access a specific document category.

    This is the atomic permission check used by the retrieval layer to
    filter chunks. It is intentionally a simple boolean — no exceptions,
    no side effects — so it can be called in a list comprehension without
    worrying about error handling.

    Args:
        username:          The username to check.
        document_category: The category tag on the document chunk.

    Returns:
        True if the user is allowed to access this category, False otherwise.
    """
    allowed = get_allowed_categories(username)
    return document_category in allowed


def get_user_role(username: str) -> str:
    """
    Return the role assigned to a user (e.g. 'finance', 'hr', 'admin').

    Used by the dashboard to determine UI visibility (admin tab gate).

    Args:
        username: The username to look up.

    Returns:
        Role string, or 'unknown' for unrecognised usernames.
    """
    users = _load_users()
    return users.get(username, {}).get("role", "unknown")


def is_admin(username: str) -> bool:
    """
    Convenience check: is this user an admin?

    Args:
        username: The username to check.

    Returns:
        True if the user's role is 'admin'.
    """
    return get_user_role(username) == "admin"