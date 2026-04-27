"""
Minimal Streamlit UI that talks to the FastAPI backend — ANNOTATED.

This is the same code as scripts/ui.py, but with comments explaining every
line and decision. After reading this, you should understand exactly what
each piece does and why it's there.

Run AFTER starting the API:
    uv run uvicorn src.api.main:app --reload

Then in another terminal:
    uv run streamlit run scripts/ui.py
"""
from __future__ import annotations

import os

# httpx is a modern requests-like HTTP client. Async-capable, but we're
# using it synchronously here. We picked it over `requests` because it's
# the same library used elsewhere in the project — one less dependency.
import httpx
import streamlit as st

# Why an env var: lets you point the UI at different API hosts without
# code changes. Useful when you containerize later (UI in one container,
# API in another, talking via service name instead of localhost).
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ────────────────────────────────────────────────────────────────────────
# Page config — must be the FIRST Streamlit call in the script
# ────────────────────────────────────────────────────────────────────────
# This sets the browser tab title, favicon, and overall layout. "wide"
# uses the full browser width, which is nicer for showing answers + sources
# side by side. The default is a narrow column.
st.set_page_config(page_title="RAG · Qdrant + Voyage + Claude", page_icon="📚", layout="wide")


# ────────────────────────────────────────────────────────────────────────
# Header
# ────────────────────────────────────────────────────────────────────────
# st.title and st.caption are the two biggest text helpers. caption is
# small grey text — perfect for "metadata" lines like the API URL.
st.title("📚 RAG playground")
st.caption(f"API: `{API_URL}` · Qdrant + Voyage + Claude")


# ────────────────────────────────────────────────────────────────────────
# Sidebar: live stats from the API
# ────────────────────────────────────────────────────────────────────────
# This calls /stats every time the page renders. With Streamlit's re-run
# model, that means every interaction triggers a stats fetch. For a learning
# app, that's fine. In production you'd cache it with @st.cache_data
# (with a TTL) to avoid hammering the API.
try:
    # 5-second timeout. If the API is down, we want to fail FAST and
    # show an error rather than freeze the page.
    stats = httpx.get(f"{API_URL}/stats", timeout=5).json()

    # st.sidebar.X puts X in the left sidebar. st.metric renders a big
    # number with a label — Streamlit's standard "dashboard widget."
    st.sidebar.metric("Indexed chunks", stats.get("vector_count", 0))
    st.sidebar.caption(f"Collection: {stats.get('collection')}")
    st.sidebar.caption(f"Dim: {stats.get('dim')}")
except Exception as e:
    # Show errors prominently rather than silently. Bare `except` is
    # justified here because we want any failure (network, JSON, etc.)
    # to surface as "API unreachable" rather than crashing the page.
    st.sidebar.error(f"API unreachable: {e}")


# ────────────────────────────────────────────────────────────────────────
# Sidebar: upload PDFs
# ────────────────────────────────────────────────────────────────────────
# st.file_uploader returns an UploadedFile (or list, with accept_multiple_files).
# Streamlit holds the file in memory until the user clicks Ingest, so we
# don't fire a network call on every rerun.
st.sidebar.divider()
st.sidebar.subheader("Upload")
uploaded = st.sidebar.file_uploader("Add a PDF", type=["pdf"], accept_multiple_files=False)
if uploaded is not None and st.sidebar.button("Ingest", type="primary"):
    with st.spinner(f"Ingesting {uploaded.name}..."):
        try:
            # httpx multipart format: {"file": (name, bytes, mime)}.
            # Matches FastAPI's UploadFile = File(...) signature.
            r = httpx.post(
                f"{API_URL}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=300,  # ingestion runs the full embed pipeline — be patient
            )
            r.raise_for_status()
            res = r.json()
            st.sidebar.success(f"✓ {res['filename']}: {res['pages']} pages, {res['chunks']} chunks")
            # Force a rerun so the stats metric and source dropdown refresh.
            st.rerun()
        except httpx.HTTPStatusError as e:
            st.sidebar.error(f"{e.response.status_code}: {e.response.text}")
        except Exception as e:
            st.sidebar.error(str(e))


# ────────────────────────────────────────────────────────────────────────
# Sidebar: source filter
# ────────────────────────────────────────────────────────────────────────
# Fetch the list of indexed sources so the user can scope queries to one
# document. "All sources" is the default — sentinel that maps to "send no
# source_filter to the API." Like /stats, this re-fetches on every rerun;
# fine for a playground.
try:
    src_list = httpx.get(f"{API_URL}/sources", timeout=5).json().get("sources", [])
except Exception:
    src_list = []

ALL_SOURCES = "All sources"
source_choice = st.sidebar.selectbox(
    "Filter by source",
    [ALL_SOURCES, *src_list],
    help="Restrict retrieval to a single PDF.",
)
source_filter = None if source_choice == ALL_SOURCES else source_choice


# ────────────────────────────────────────────────────────────────────────
# Chat history state
# ────────────────────────────────────────────────────────────────────────
# Streamlit re-executes the whole script on every interaction, so plain
# Python variables reset each run. st.session_state is the escape hatch:
# a dict-like object that survives across reruns within one browser tab.
#
# Each message is {"role": "user"|"assistant", "content": str,
# "sources": list | None}. Storing sources next to the answer lets us
# re-render their expanders on every rerun without re-calling the API.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls. top_k applies to all turns, so it belongs here
# rather than next to the chat input.
top_k = st.sidebar.number_input("top_k", min_value=1, max_value=20, value=5)
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()


# ────────────────────────────────────────────────────────────────────────
# Replay history
# ────────────────────────────────────────────────────────────────────────
# st.chat_message renders a styled bubble keyed by role — "user" and
# "assistant" get distinct avatars and alignment automatically. We
# replay every prior turn on each rerun so the conversation stays visible.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            for s in msg["sources"]:
                with st.expander(f"[{s['label']}] {s['source']} · page {s['page']} · score {s['score']}"):
                    st.write(s["preview"] + "…")


# ────────────────────────────────────────────────────────────────────────
# Chat input — pinned to the bottom of the page
# ────────────────────────────────────────────────────────────────────────
# st.chat_input is purpose-built for chat UIs: anchored to the bottom,
# returns the submitted text on the run after submit, None otherwise.
# It also handles the "don't rerun on every keystroke" problem that
# st.form solved before — only submit triggers a rerun.
if q := st.chat_input("Ask a question about your PDFs"):
    # Annotate the user message with the active filter so history shows
    # what scope the question was answered against.
    user_content = q if source_filter is None else f"_(source: `{source_filter}`)_  \n{q}"
    st.session_state.messages.append({"role": "user", "content": user_content})
    with st.chat_message("user"):
        st.markdown(user_content)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            try:
                # 60s timeout because Claude generation can take a few
                # seconds — patient, but bounded.
                payload = {"question": q, "top_k": int(top_k)}
                if source_filter:
                    payload["source_filter"] = source_filter
                r = httpx.post(f"{API_URL}/query", json=payload, timeout=60)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPStatusError as e:
                st.error(f"{e.response.status_code}: {e.response.text}")
                st.stop()
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.markdown(data["answer"])
        for s in data["sources"]:
            with st.expander(f"[{s['label']}] {s['source']} · page {s['page']} · score {s['score']}"):
                st.write(s["preview"] + "…")

    # Persist the assistant turn so it survives the next rerun.
    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"],
        "sources": data["sources"],
    })


# ────────────────────────────────────────────────────────────────────────
# What's missing (and worth building next)
# ────────────────────────────────────────────────────────────────────────
# 1. STREAMING: Streamlit has st.write_stream which can consume an
#    SSE response from a /query/stream endpoint. Adds the typing-style
#    UX you saw in the generation walkthrough. Phase 2 territory.
#
# 2. CONVERSATIONAL CONTEXT: Right now the API answers each turn
#    standalone — prior chat history isn't sent. To support follow-ups
#    like "what about page 3?", forward st.session_state.messages as
#    a `history` field on /query and feed it into the prompt.