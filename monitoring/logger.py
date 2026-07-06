"""
monitoring/logger.py

Persistent query logging to SQLite for audit trails, security monitoring,
and performance analytics.

Design decisions worth understanding:

1. Connection-per-operation pattern (not a persistent connection):
   Each function opens, uses, and closes its own connection. This is
   safer than a module-level persistent connection in a Streamlit app
   where multiple browser sessions may write concurrently. SQLite handles
   concurrent reads fine; concurrent writes are serialized by the OS-level
   file lock. Connection-per-operation ensures no session holds a lock
   longer than a single query.

2. Row factory (sqlite3.Row):
   By default sqlite3 returns tuples. Setting row_factory=sqlite3.Row
   gives us dict-like row objects: row['username'] instead of row[3].
   We convert to plain dicts before returning so callers don't need to
   know about sqlite3.Row internals.

3. JSON for citations:
   SQLite has no native array type. Citations (a list of dicts) are
   serialized to JSON string on write and deserialized on read.
   SQLite 3.38+ has JSON functions but we keep it explicit for clarity.

4. ISO 8601 timestamps:
   Stored as TEXT in 'YYYY-MM-DDTHH:MM:SS.ffffff' format. Lexicographic
   sort = chronological sort. Works correctly in ORDER BY without any
   date parsing. Universally parseable by pandas, JavaScript, etc.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from config import DB_PATH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS queries (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        TEXT    NOT NULL,
    username         TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    response         TEXT,
    latency          REAL,
    input_tokens     INTEGER DEFAULT 0,
    output_tokens    INTEGER DEFAULT 0,
    injection_flag   INTEGER DEFAULT 0,
    toxicity_flag    INTEGER DEFAULT 0,
    access_denied    INTEGER DEFAULT 0,
    citations        TEXT    DEFAULT '[]',
    injection_reason TEXT,
    toxicity_score   REAL    DEFAULT 0.0
);
"""


def _get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection with dict-like row access.

    Always use this inside a 'with' block so the connection closes
    and commits/rolls back automatically:

        with _get_connection() as conn:
            conn.execute(...)

    The 'with' block on a sqlite3 connection commits on exit if no
    exception occurred, rolls back if an exception was raised.
    """
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create the queries table if it does not already exist.

    Safe to call multiple times — CREATE TABLE IF NOT EXISTS is idempotent.
    Call this once at application startup before any logging occurs.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
    print(f"[LOGGER] Database ready: {DB_PATH}")


def log_query(data: dict) -> int:
    """
    Insert a query record into the database.

    Accepts a flat dict — all fields are optional except 'username'
    and 'query'. Missing fields default to safe zero/empty values so
    callers can log partial records (e.g. access-denied queries that
    never reach the LLM and have no latency or token counts).

    Args:
        data: Dict with any of these keys:
            username        (str)   : required
            query           (str)   : required
            response        (str)   : LLM answer text
            latency         (float) : API call duration in seconds
            input_tokens    (int)   : prompt token count
            output_tokens   (int)   : completion token count
            injection_flag  (bool)  : True if injection detected
            toxicity_flag   (bool)  : True if toxicity detected
            access_denied   (bool)  : True if access control blocked
            citations       (list)  : list of citation dicts
            injection_reason(str)   : reason string from guardrail
            toxicity_score  (float) : raw toxicity probability

    Returns:
        The integer row ID of the inserted record.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO queries (
                timestamp, username, query, response,
                latency, input_tokens, output_tokens,
                injection_flag, toxicity_flag, access_denied,
                citations, injection_reason, toxicity_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                data.get("username", "unknown"),
                data.get("query", ""),
                data.get("response", ""),
                data.get("latency", 0.0),
                data.get("input_tokens", 0),
                data.get("output_tokens", 0),
                int(data.get("injection_flag", False)),
                int(data.get("toxicity_flag", False)),
                int(data.get("access_denied", False)),
                json.dumps(data.get("citations", [])),
                data.get("injection_reason", ""),
                data.get("toxicity_score", 0.0),
            ),
        )
        return cursor.lastrowid


def get_all_logs() -> List[dict]:
    """
    Return all query records, newest first.

    Used by the admin panel to display the full query log table.
    Citations are deserialized back to Python lists.

    Returns:
        List of dicts, ordered by timestamp descending.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC"
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_flagged_logs() -> List[dict]:
    """
    Return only queries that triggered at least one security flag.

    A query is flagged if any of injection_flag, toxicity_flag, or
    access_denied is set to 1. Used by the admin security panel.

    Returns:
        List of flagged query dicts, newest first.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            """
            SELECT * FROM queries
            WHERE injection_flag = 1
               OR toxicity_flag  = 1
               OR access_denied  = 1
            ORDER BY timestamp DESC
            """
        ).fetchall()

    return [_row_to_dict(row) for row in rows]


def get_stats() -> dict:
    """
    Compute aggregate statistics for the admin dashboard metrics cards.

    Returns a single dict so the dashboard only needs one function call
    to populate all four metric cards (total, flagged, avg latency,
    most active user) and the queries-per-day chart data.

    Returns:
        Dict with keys:
            total_queries    (int)
            flagged_queries  (int)
            avg_latency      (float) seconds
            most_active_user (str)
            queries_per_day  (list of {date: str, count: int})
    """
    with _get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]

        flagged = conn.execute(
            """SELECT COUNT(*) FROM queries
               WHERE injection_flag=1 OR toxicity_flag=1 OR access_denied=1"""
        ).fetchone()[0]

        avg_latency_row = conn.execute(
            "SELECT AVG(latency) FROM queries WHERE latency > 0"
        ).fetchone()[0]
        avg_latency = round(avg_latency_row or 0.0, 3)

        top_user_row = conn.execute(
            """SELECT username, COUNT(*) as cnt FROM queries
               GROUP BY username ORDER BY cnt DESC LIMIT 1"""
        ).fetchone()
        most_active_user = top_user_row[0] if top_user_row else "—"

        daily_rows = conn.execute(
            """SELECT DATE(timestamp) as date, COUNT(*) as count
               FROM queries
               GROUP BY DATE(timestamp)
               ORDER BY date DESC
               LIMIT 30"""
        ).fetchall()
        queries_per_day = [
            {"date": row["date"], "count": row["count"]}
            for row in daily_rows
        ]

    return {
        "total_queries": total,
        "flagged_queries": flagged,
        "avg_latency": avg_latency,
        "most_active_user": most_active_user,
        "queries_per_day": queries_per_day,
    }


def _row_to_dict(row: sqlite3.Row) -> dict:
    """
    Convert a sqlite3.Row to a plain Python dict.

    Also deserializes the citations JSON string back to a Python list.
    Kept private — an implementation detail of how we store citations.

    Args:
        row: A sqlite3.Row object from any SELECT query.

    Returns:
        Plain dict with all columns, citations as a Python list.
    """
    d = dict(row)
    try:
        d["citations"] = json.loads(d.get("citations") or "[]")
    except (json.JSONDecodeError, TypeError):
        d["citations"] = []
    return d