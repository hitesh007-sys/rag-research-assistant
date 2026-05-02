# app.py
import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_loader import load_and_chunk_pdf, load_and_chunk_multiple_pdfs
from utils.embeddings import (
    create_vector_store,
    add_to_vector_store,
    load_vector_store,
    vector_store_exists,
    get_vector_store_sources
)
from utils.qa_chain import (
    build_qa_chain,
    get_answer,
    build_conversational_chain,
    get_conversational_answer,
    get_streaming_answer
)
from utils.history_manager import (
    add_to_history,
    load_history,
    remove_from_history,
    clear_history,
    get_history_stats
)
from utils.reranker import is_reranker_available

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f77b4;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1rem;
    color: #888;
    margin-bottom: 2rem;
}
.answer-box {
    background-color: #f0f7ff;
    border-left: 4px solid #1f77b4;
    padding: 1rem 1.5rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
    color: #1a1a1a !important;
    font-size: 1rem;
    line-height: 1.6;
}
.source-box {
    background-color: #f9f9f9;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #1a1a1a !important;
    line-height: 1.5;
}
.status-success { color: #28a745; font-weight: 600; }
.status-error   { color: #dc3545; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session_state():
    # Check reranker availability once at startup
    reranker_ok = is_reranker_available()

    defaults = {
        "qa_chain":      None,
        "conv_chain":    None,
        "vector_store":  None,
        "chat_history":  [],
        "pdf_processed": False,
        "pdf_name":      "",
        "initialized":   False,
        "use_hybrid":    True,
        "use_reranking": reranker_ok,   # auto-disable on cloud if unavailable
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Build chains ONCE after PDF is processed ──────────────────────────────────
def setup_chains(vector_store):
    """Called once after a new PDF is processed or existing store is loaded."""
    use_hybrid = st.session_state.get("use_hybrid", True)
    st.session_state.vector_store  = vector_store
    st.session_state.qa_chain      = build_qa_chain(vector_store)
    st.session_state.conv_chain    = build_conversational_chain(
        vector_store, use_hybrid=use_hybrid
    )
    st.session_state.pdf_processed = True
    st.session_state.initialized   = True


# ── PDF processing ────────────────────────────────────────────────────────────
def process_pdf(uploaded_file, add_to_existing: bool = False):
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            chunks = load_and_chunk_pdf(tmp_path)
            os.unlink(tmp_path)

            if add_to_existing and vector_store_exists():
                vector_store = add_to_vector_store(chunks)
            else:
                vector_store = create_vector_store(chunks)

            setup_chains(vector_store)
            st.session_state.pdf_name     = uploaded_file.name
            st.session_state.chat_history = []

            add_to_history(
                uploaded_file.name,
                len(chunks),
                uploaded_file.size
            )

            sources = get_vector_store_sources()
            st.success(
                f"✅ Processed **{uploaded_file.name}** "
                f"→ **{len(chunks)} chunks**  \n"
                f"📚 Active documents: **{', '.join(sources)}**"
            )

        except Exception as e:
            st.error(f"❌ {str(e)}")


# ── Load existing vector store ────────────────────────────────────────────────
def load_existing():
    with st.spinner("Loading existing data..."):
        try:
            vector_store = load_vector_store()
            setup_chains(vector_store)
            st.session_state.pdf_name = "Previously processed document"
            st.success("✅ Loaded existing vector store!")
        except Exception as e:
            st.error(f"❌ {str(e)}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Brand header
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #1f77b4, #0d47a1);
            padding: 1rem 1.2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
        ">
            <div style="font-size:1.3rem;margin-bottom:0.3rem;">📄</div>
            <div style="color:white;font-size:1rem;font-weight:700;
                        letter-spacing:0.05em;line-height:1.3;">
                RAG RESEARCH<br>ASSISTANT
            </div>
            <div style="color:#90caf9;font-size:0.7rem;margin-top:0.3rem;
                        letter-spacing:0.08em;">
                powered by Groq × LangChain
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("## 📄 Upload Document")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload any PDF document"
        )

        if uploaded_file:
            st.info(f"Selected: **{uploaded_file.name}**")

            sources = get_vector_store_sources()
            if sources:
                st.markdown("**Currently loaded:**")
                for src in sources:
                    st.markdown(f"- `{src}`")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(
                    "Process PDF",
                    type="primary",
                    use_container_width=True,
                    help="Replace current data with this PDF"
                ):
                    process_pdf(uploaded_file, add_to_existing=False)

            with col2:
                if st.button(
                    "Add to existing",
                    use_container_width=True,
                    disabled=not vector_store_exists(),
                    help="Add this PDF to existing documents"
                ):
                    process_pdf(uploaded_file, add_to_existing=True)

        st.markdown("---")

        # Status
        if st.session_state.pdf_processed:
            st.markdown(
                f'<p class="status-success">'
                f'✅ Ready: {st.session_state.pdf_name}</p>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<p class="status-error">⚠️ No document loaded</p>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Load existing
        if vector_store_exists():
            st.markdown("### Previously Processed")
            if st.button("Load Existing Data", use_container_width=True):
                load_existing()

        # Clear history
        if st.session_state.chat_history:
            if st.button("Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                if st.session_state.vector_store:
                    st.session_state.conv_chain = build_conversational_chain(
                        st.session_state.vector_store,
                        use_hybrid=st.session_state.use_hybrid
                    )
                st.rerun()

        st.markdown("---")

        # ── Upload history ────────────────────────────────────────────────────
        history = load_history()
        stats   = get_history_stats()

        with st.expander(
            f"📁 Document History ({stats['total_docs']} files)",
            expanded=False
        ):
            if not history:
                st.caption("No documents processed yet.")
            else:
                st.markdown(
                    f"**Total docs:** {stats['total_docs']}  \n"
                    f"**Total chunks:** {stats['total_chunks']}  \n"
                    f"**Total size:** {stats['total_size_kb']} KB"
                )
                st.markdown("---")

                for entry in reversed(history):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(
                            f"**{entry['filename']}**  \n"
                            f"Chunks: `{entry['chunks']}` · "
                            f"Size: `{entry['file_size_kb']} KB`  \n"
                            f"Last: `{entry['last_processed']}`"
                        )

                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"del_{entry['filename']}",
                            help=f"Remove {entry['filename']} from history"
                        ):
                            remove_from_history(entry["filename"])
                            st.rerun()

                st.markdown("---")
                if st.button("Clear All History", use_container_width=True):
                    clear_history()
                    st.rerun()

        # ── Settings ──────────────────────────────────────────────────────────
        with st.expander("⚙️ Settings", expanded=False):
            st.markdown(
                f"**Model**  \n"
                f"`{os.getenv('GROQ_MODEL_NAME','llama-3.3-70b-versatile')}`"
            )
            st.markdown(
                f"**Chunk size**  \n`{os.getenv('CHUNK_SIZE','500')}`"
            )
            st.markdown(
                f"**Top-k chunks**  \n`{os.getenv('RETRIEVAL_K','4')}`"
            )
            st.markdown("**Embedding model**  \n`all-MiniLM-L6-v2`")
            st.markdown("**Vector store**  \n`ChromaDB (local)`")
            st.markdown("**Memory**  \n`Last 5 exchanges`")

            st.markdown("---")

            # ── Hybrid search toggle ──────────────────────────────────────────
            if "use_hybrid" not in st.session_state:
                st.session_state.use_hybrid = True

            use_hybrid = st.toggle(
                "Hybrid Search (BM25 + Semantic)",
                value=st.session_state.use_hybrid,
                help="Combines keyword and semantic search for better results"
            )

            if use_hybrid:
                st.success("Hybrid search ON ✅")
                st.caption("BM25(40%) + Semantic(60%)")
            else:
                st.warning("Semantic search only")
                st.caption("Pure vector similarity")

            st.markdown("---")

            # ── Reranking toggle (cloud-safe) ─────────────────────────────────
            reranker_supported = is_reranker_available()

            if reranker_supported:
                if "use_reranking" not in st.session_state:
                    st.session_state.use_reranking = True

                use_reranking = st.toggle(
                    "Reranking (Cross-encoder)",
                    value=st.session_state.use_reranking,
                    help="Re-scores chunks for higher accuracy. "
                         "Adds ~1-2 seconds per query."
                )

                if use_reranking:
                    st.success("Reranking ON ✅")
                    st.caption("cross-encoder/ms-marco-MiniLM-L-6-v2")
                else:
                    st.warning("Reranking OFF")
                    st.caption("Chunks used in retrieval order")
            else:
                # Gracefully disable on Streamlit Cloud
                use_reranking = False
                st.session_state.use_reranking = False
                st.info("⚠️ Reranking unavailable on this platform")
                st.caption("Run locally for reranking support")

            # Sync env var
            os.environ["USE_RERANKING"] = (
                "true" if use_reranking else "false"
            )

            # Rebuild chain only when toggle actually changes
            hybrid_changed    = use_hybrid    != st.session_state.use_hybrid
            reranking_changed = use_reranking != st.session_state.use_reranking

            st.session_state.use_hybrid    = use_hybrid
            st.session_state.use_reranking = use_reranking

            if (hybrid_changed or reranking_changed) and \
                    st.session_state.vector_store is not None:
                with st.spinner("Switching retriever..."):
                    st.session_state.conv_chain = build_conversational_chain(
                        st.session_state.vector_store,
                        use_hybrid=use_hybrid
                    )
                st.rerun()


# ── Chat area ─────────────────────────────────────────────────────────────────
def render_chat():
    st.markdown(
        '<p class="main-header">📄 RAG Research Assistant</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload a PDF and ask questions — '
        'answers are grounded in your document with source citations.</p>',
        unsafe_allow_html=True
    )

    # Render chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(
                f'<div class="answer-box">{entry["answer"]}</div>',
                unsafe_allow_html=True
            )
            if entry.get("sources"):
                with st.expander(
                    f"📚 View {len(entry['sources'])} source chunk(s)"
                ):
                    for i, src in enumerate(entry["sources"]):
                        src_file = src.metadata.get("source", "Unknown")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i+1}</strong> '
                            f'<span style="color:#1f77b4;font-size:0.8rem;">'
                            f'📄 {src_file}</span><br><br>'
                            f'{src.page_content[:400]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

    # Input
    if st.session_state.pdf_processed and st.session_state.conv_chain:
        question = st.chat_input("Ask a question about your document...")
        if question:
            handle_question(question)
    else:
        st.info("👈 Upload and process a PDF using the sidebar to start.")


# ── Handle a question ─────────────────────────────────────────────────────────
def handle_question(question: str):
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            full_answer        = ""
            answer_placeholder = st.empty()

            for token in get_streaming_answer(
                st.session_state.conv_chain, question
            ):
                full_answer += token
                answer_placeholder.markdown(
                    f'<div class="answer-box">{full_answer}▌</div>',
                    unsafe_allow_html=True
                )

            # Final answer without cursor
            answer_placeholder.markdown(
                f'<div class="answer-box">{full_answer}</div>',
                unsafe_allow_html=True
            )

            # Get sources for citations
            retriever = st.session_state.conv_chain.retriever
            sources   = retriever.invoke(question)

            if sources:
                with st.expander(
                    f"📚 View {len(sources)} source chunk(s)"
                ):
                    for i, src in enumerate(sources):
                        src_file = src.metadata.get("source", "Unknown")
                        st.markdown(
                            f'<div class="source-box">'
                            f'<strong>Source {i+1}</strong> '
                            f'<span style="color:#1f77b4;font-size:0.8rem;">'
                            f'📄 {src_file}</span><br><br>'
                            f'{src.page_content[:400]}...'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            st.session_state.chat_history.append({
                "question": question,
                "answer":   full_answer,
                "sources":  sources
            })

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()