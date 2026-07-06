"""
dashboard/app.py

Streamlit dashboard for the Secure RAG System.
Two tabs: Chat (all users) and Admin (admin only, password-gated).

Key Streamlit concepts used here:
    st.session_state  — persists data across reruns (chat history, user)
    @st.cache_resource — caches expensive objects (models, index) once
                         per process, not per rerun
    st.rerun()        — explicitly triggers a fresh rerun after state changes
"""

import json
import sys
from pathlib import Path

import streamlit as st
import pandas as pd

# Ensure project root is on sys.path when running via `streamlit run`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ADMIN_PASSWORD, validate_config
from ingestion.document_loader import load_pdf
from ingestion.text_splitter import split_documents
from ingestion.embedder import embed_and_store
from retrieval.retriever import retrieve
from guardrails.prompt_injection import detect_injection
from guardrails.toxicity import detect_toxicity
from guardrails.access_control import get_allowed_categories, is_admin
from generation.prompt_builder import build_prompt
from generation.llm_caller import generate_response
from monitoring.logger import init_db, log_query, get_all_logs, get_flagged_logs, get_stats

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Secure RAG System",
    page_icon="🔒",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Initialisation — runs once per session
# ---------------------------------------------------------------------------
validate_config()
init_db()

# Initialise session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []      # chat history
if "username" not in st.session_state:
    st.session_state.username = "alice"
if "admin_unlocked" not in st.session_state:
    st.session_state.admin_unlocked = False


# ---------------------------------------------------------------------------
# Cached resource loaders
# @st.cache_resource caches the return value for the lifetime of the process.
# Without this, the FAISS index and all three ML models would reload on every
# single user interaction — turning a 1s response into a 30s one.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_retriever_resources():
    """Load and cache FAISS index + embedding model."""
    from ingestion.embedder import load_index
    from sentence_transformers import SentenceTransformer
    from config import EMBEDDING_MODEL_NAME
    index, metadata = load_index()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return index, metadata, model


@st.cache_resource
def load_toxicity_model():
    """Load and cache the toxicity classifier."""
    from transformers import pipeline
    from config import TOXICITY_MODEL_NAME
    return pipeline("text-classification", model=TOXICITY_MODEL_NAME)


# Warm up models on first load so the first query isn't slow
try:
    load_retriever_resources()
    load_toxicity_model()
except FileNotFoundError:
    pass  # No index yet — user needs to upload a PDF first


# ---------------------------------------------------------------------------
# Helper: run the full RAG pipeline on a query
# ---------------------------------------------------------------------------
def run_pipeline(query: str, username: str) -> dict:
    """
    Execute guardrails → retrieval → generation → logging.

    Returns a result dict consumed by the chat UI to render
    the response, citations, and any security warnings.
    """
    result = {
        "query": query,
        "username": username,
        "response": "",
        "citations": [],
        "latency": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "injection_flag": False,
        "toxicity_flag": False,
        "access_denied": False,
        "injection_reason": "",
        "toxicity_score": 0.0,
        "blocked": False,
        "block_reason": "",
    }

    # --- Guardrail 1: Prompt injection ---
    injection_result = detect_injection(query)
    result["injection_flag"] = injection_result["is_injection"]
    result["injection_reason"] = injection_result["reason"]

    if injection_result["is_injection"]:
        result["blocked"] = True
        result["block_reason"] = f"🚨 Prompt injection detected: {injection_result['reason']}"
        result["response"] = "Your query was blocked: potential prompt injection attempt detected."
        log_query(result)
        return result

    # --- Guardrail 2: Toxicity ---
    toxicity_result = detect_toxicity(query)
    result["toxicity_flag"] = toxicity_result["is_toxic"]
    result["toxicity_score"] = toxicity_result["score"]

    if toxicity_result["is_toxic"]:
        result["blocked"] = True
        result["block_reason"] = f"🚨 Toxic content detected (score: {toxicity_result['score']:.2f})"
        result["response"] = "Your query was blocked: toxic content detected."
        log_query(result)
        return result

    # --- Guardrail 3: Retrieval with access control ---
    chunks = retrieve(query, username, top_k=5)

    if not chunks:
        result["access_denied"] = True
        result["blocked"] = True
        result["block_reason"] = "🔒 Access denied: no documents available for your access level."
        result["response"] = "You don't have access to documents relevant to this query."
        log_query(result)
        return result

    # --- Generation ---
    prompt_data = build_prompt(query, chunks, username)
    llm_result = generate_response(prompt_data["messages"])

    result.update({
        "response": llm_result["response"],
        "citations": llm_result["citations"],
        "latency": llm_result["latency"],
        "input_tokens": llm_result["input_tokens"],
        "output_tokens": llm_result["output_tokens"],
    })

    log_query(result)
    return result


# ---------------------------------------------------------------------------
# Helper: ingest an uploaded PDF
# ---------------------------------------------------------------------------
def ingest_uploaded_pdf(uploaded_file, category: str) -> bool:
    """
    Save uploaded PDF to data/documents/ and re-run ingestion pipeline.
    Returns True on success, False on failure.
    """
    from config import DOCUMENTS_DIR

    # Prefix filename with category for access control inference
    safe_name = f"{category}_{uploaded_file.name}"
    dest = DOCUMENTS_DIR / safe_name

    try:
        dest.write_bytes(uploaded_file.read())
        docs = load_pdf(dest)
        chunks = split_documents(docs)
        embed_and_store(chunks)
        # Clear cache so retriever picks up the new index
        load_retriever_resources.clear()
        return True
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return False


# ---------------------------------------------------------------------------
# TAB 1 — CHAT
# ---------------------------------------------------------------------------
def render_chat_tab():
    st.title("🔒 Secure RAG System")
    st.caption("Ask questions about your documents. Queries are security-checked before retrieval.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        # Username selector
        new_user = st.selectbox(
            "Logged in as",
            ["alice", "bob", "admin"],
            index=["alice", "bob", "admin"].index(st.session_state.username),
        )
        if new_user != st.session_state.username:
            st.session_state.username = new_user
            st.session_state.messages = []  # clear chat on user switch
            st.rerun()

        # Show access level
        categories = get_allowed_categories(st.session_state.username)
        st.caption(f"Access: `{'`, `'.join(categories)}`")

        st.divider()

        # PDF uploader (admin only)
        if is_admin(st.session_state.username):
            st.subheader("Upload Document")
            category = st.selectbox("Document category", ["finance", "hr", "general"])
            uploaded = st.file_uploader("Choose a PDF", type="pdf")
            if uploaded and st.button("Ingest Document"):
                with st.spinner("Ingesting..."):
                    success = ingest_uploaded_pdf(uploaded, category)
                if success:
                    st.success(f"Ingested: {uploaded.name}")
                else:
                    st.error("Ingestion failed. Check terminal for details.")
        else:
            st.info("Document upload is available to admin users only.")

        st.divider()
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    # --- Chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("blocked"):
                st.error(msg["block_reason"])
            st.write(msg["content"])
            if msg.get("citations"):
                with st.expander("📄 Sources"):
                    for c in msg["citations"]:
                        st.write(f"• **{c['filename']}** — Page {c['page']}")
            if msg.get("latency"):
                st.caption(
                    f"⏱ {msg['latency']}s · "
                    f"↑{msg.get('input_tokens',0)} tokens · "
                    f"↓{msg.get('output_tokens',0)} tokens"
                )

    # --- Chat input ---
    if prompt := st.chat_input("Ask a question about your documents..."):

        # Show user message
        with st.chat_message("user"):
            st.write(prompt)

        # Run pipeline and show response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_pipeline(prompt, st.session_state.username)

            if result["blocked"]:
                st.error(result["block_reason"])

            st.write(result["response"])

            if result["citations"]:
                with st.expander("📄 Sources"):
                    for c in result["citations"]:
                        st.write(f"• **{c['filename']}** — Page {c['page']}")

            if result["latency"] > 0:
                st.caption(
                    f"⏱ {result['latency']}s · "
                    f"↑{result['input_tokens']} tokens · "
                    f"↓{result['output_tokens']} tokens"
                )

        # Persist to session state
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "blocked": result["blocked"],
            "block_reason": result.get("block_reason", ""),
            "citations": result["citations"],
            "latency": result["latency"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
        })


# ---------------------------------------------------------------------------
# TAB 2 — ADMIN
# ---------------------------------------------------------------------------
def render_admin_tab():
    st.title("🛡️ Admin Monitoring Panel")

    # Password gate
    if not st.session_state.admin_unlocked:
        pwd = st.text_input("Enter admin password", type="password")
        if st.button("Unlock"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_unlocked = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return

    # --- Metrics ---
    stats = get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Queries", stats["total_queries"])
    c2.metric("Flagged Queries", stats["flagged_queries"])
    c3.metric("Avg Latency", f"{stats['avg_latency']}s")
    c4.metric("Most Active User", stats["most_active_user"])

    st.divider()

    # --- Queries per day chart ---
    st.subheader("Queries Per Day")
    if stats["queries_per_day"]:
        df_daily = pd.DataFrame(stats["queries_per_day"])
        df_daily = df_daily.set_index("date").sort_index()
        st.bar_chart(df_daily["count"])
    else:
        st.info("No query data yet.")

    st.divider()

    # --- Security flags breakdown ---
    st.subheader("Security Flags Breakdown")
    all_logs = get_all_logs()
    if all_logs:
        df = pd.DataFrame(all_logs)
        flag_counts = {
            "Injection": int(df["injection_flag"].sum()),
            "Toxicity": int(df["toxicity_flag"].sum()),
            "Access Denied": int(df["access_denied"].sum()),
        }
        df_flags = pd.DataFrame(
            list(flag_counts.items()),
            columns=["Flag Type", "Count"]
        ).set_index("Flag Type")
        st.bar_chart(df_flags["Count"])
    else:
        st.info("No data yet.")

    st.divider()

    # --- Full query log table ---
    st.subheader("Query Log")
    view = st.radio("Show", ["All queries", "Flagged only"], horizontal=True)
    logs = get_flagged_logs() if view == "Flagged only" else get_all_logs()

    if not logs:
        st.info("No records to display.")
        return

    df_logs = pd.DataFrame(logs)

    # Select and rename columns for display
    display_cols = {
        "timestamp": "Timestamp",
        "username": "User",
        "query": "Query",
        "response": "Response",
        "latency": "Latency (s)",
        "injection_flag": "Injection",
        "toxicity_flag": "Toxic",
        "access_denied": "Denied",
        "toxicity_score": "Tox Score",
    }
    df_display = df_logs[[c for c in display_cols if c in df_logs.columns]].copy()
    df_display.columns = [display_cols[c] for c in df_display.columns]

    # Highlight flagged rows in red
    def highlight_flagged(row):
        is_flagged = row.get("Injection") or row.get("Toxic") or row.get("Denied")
        return ["background-color: #ffcccc" if is_flagged else "" for _ in row]

    st.dataframe(
        df_display.style.apply(highlight_flagged, axis=1),
        use_container_width=True,
        height=400,
    )

    st.caption(f"Showing {len(logs)} record(s)")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
if is_admin(st.session_state.username):
    tab1, tab2 = st.tabs(["💬 Chat", "🛡️ Admin"])
    with tab1:
        render_chat_tab()
    with tab2:
        render_admin_tab()
else:
    render_chat_tab()