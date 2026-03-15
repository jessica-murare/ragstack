# app.py
import streamlit as st
import sys
from pathlib import Path
import tempfile
import os

sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="RAGStack",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAGStack")
st.caption("Upload documents, ask questions, get grounded answers.")

# --- Init pipeline (cached so it doesn't reload on every interaction) ---
@st.cache_resource
def load_pipeline():
    return RAGPipeline()

rag = load_pipeline()

# --- Sidebar: Document Upload ---
with st.sidebar:
    st.header("Documents")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs or Markdown files",
        type=["pdf", "md"],
        accept_multiple_files=True,
    )

    # URL input
    url_input = st.text_area(
        "Or paste URLs (one per line)",
        placeholder="https://example.com/article",
        height=100,
    )

    index_btn = st.button("Index Documents", type="primary", use_container_width=True)

    if index_btn:
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]
        saved_paths = []

        # Save uploaded files to data/raw/
        if uploaded_files:
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)

            for f in uploaded_files:
                dest = raw_dir / f.name
                dest.write_bytes(f.read())
                saved_paths.append(str(dest))
                st.sidebar.success(f"Saved: {f.name}")

        with st.spinner("Indexing documents..."):
            rag.index(urls=urls if urls else None)

        st.sidebar.success(f"Indexed {rag.store.count()} chunks!")

    # Show current index stats
    st.divider()
    st.metric("Chunks in index", rag.store.count())

# --- Main: Chat Interface ---
st.header("Ask a Question")

# Keep chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(src)

# Chat input
if question := st.chat_input("Ask something about your documents..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Generate answer
    with st.chat_message("assistant"):
        if rag.store.count() == 0:
            answer = "Please upload and index some documents first."
            st.warning(answer)
            sources = []
        else:
            with st.spinner("Thinking..."):
                result = rag.query(question)
            answer = result["answer"]
            sources = result["sources"]

            st.write(answer)

            col1, col2 = st.columns([3, 1])
            with col2:
                st.caption(f"Chunks used: {result['num_chunks']}")

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(src)

    # Save assistant message to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })