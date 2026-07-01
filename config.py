"""
config.py

Single source of truth for the Secure RAG System's configuration.
Every other module imports constants from here instead of calling
os.getenv() directly — this keeps secrets and tunable parameters
in one place and makes the whole system easier to reconfigure.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
USERS_FILE = BASE_DIR / "users.json"
DB_PATH = BASE_DIR / "data" / "rag_logs.db"

# Ensure runtime data folders exist even on a fresh clone
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM (Groq)
# ---------------------------------------------------------------------------
# Free-tier API: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Used for the actual grounded answer generation (Phase 5)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Used as the cheap/fast "LLM-as-judge" for prompt injection scoring (Phase 4)
GROQ_JUDGE_MODEL = os.getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # output size of all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K_RESULTS = 5

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
TOXICITY_MODEL_NAME = "martin-ha/toxic-comment-model"
TOXICITY_THRESHOLD = 0.7
INJECTION_LLM_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


def validate_config() -> None:
    """
    Fail fast and loudly if required secrets are missing.

    Call this once at startup (main.py / dashboard/app.py) instead of
    letting a missing key surface as a confusing error deep inside an
    API call.
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not ADMIN_PASSWORD:
        missing.append("ADMIN_PASSWORD")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file (see .env for the expected keys)."
        )
