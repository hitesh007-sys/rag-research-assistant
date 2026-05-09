# app.py
import os
import tempfile
from datetime import datetime
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
from utils.paper_metadata import process_multiple_papers
from utils.review_chain import generate_full_review
from utils.report_exporter import export_to_docx, export_to_pdf, generate_filename
from utils.comparison_chain import generate_full_comparison, comparison_to_table_rows
from utils.comparison_exporter import export_comparison_to_docx, export_comparison_to_pdf, generate_comparison_filename

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
        "use_reranking": reranker_ok,
        "review_data":   None,
        "review_papers": [],
        "review_topic":  "",
        "comparison_data":   None,     # ← add
        "comparison_papers": [],       # ← add
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Build chains ONCE ─────────────────────────────────────────────────────────
def setup_chains(vector_store):
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

        if vector_store_exists():
            st.markdown("### Previously Processed")
            if st.button("Load Existing Data", use_container_width=True):
                load_existing()

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

            reranker_supported = is_reranker_available()

            if reranker_supported:
                if "use_reranking" not in st.session_state:
                    st.session_state.use_reranking = True

                use_reranking = st.toggle(
                    "Reranking (Cross-encoder)",
                    value=st.session_state.use_reranking,
                    help="Re-scores chunks for higher accuracy."
                )

                if use_reranking:
                    st.success("Reranking ON ✅")
                    st.caption("cross-encoder/ms-marco-MiniLM-L-6-v2")
                else:
                    st.warning("Reranking OFF")
                    st.caption("Chunks used in retrieval order")
            else:
                use_reranking = False
                st.session_state.use_reranking = False
                st.info("⚠️ Reranking unavailable on this platform")
                st.caption("Run locally for reranking support")

            os.environ["USE_RERANKING"] = (
                "true" if use_reranking else "false"
            )

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

            answer_placeholder.markdown(
                f'<div class="answer-box">{full_answer}</div>',
                unsafe_allow_html=True
            )

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


# ── Literature Review Tab ─────────────────────────────────────────────────────
def render_literature_review_tab():
    st.markdown(
        '<p class="main-header">📝 Auto Literature Review</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload 2–10 research papers and generate '
        'a complete structured literature review with one click. '
        'Exports to Word and PDF.</p>',
        unsafe_allow_html=True
    )

    # ── Step 1: Upload papers ─────────────────────────────────
    st.markdown("### Step 1 — Upload research papers")
    st.caption(
        "Upload 2–10 PDF papers on the same topic. "
        "The more papers, the richer the review."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="lit_review_uploader",
        help="Upload multiple research papers at once"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} file(s) selected:")
        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(f"  - `{f.name}` ({size_kb} KB)")

    st.markdown("---")

    # ── Step 2: Topic input ───────────────────────────────────
    st.markdown("### Step 2 — Review topic (optional)")
    st.caption(
        "Leave blank to auto-detect from papers. "
        "Or specify e.g. 'Transformer models in NLP'"
    )

    topic = st.text_input(
        "Review topic",
        value=st.session_state.get("review_topic", ""),
        placeholder="e.g. Large Language Models, Medical Image Segmentation...",
        key="review_topic_input"
    )

    st.markdown("---")

    # ── Step 3: Generate button ───────────────────────────────
    st.markdown("### Step 3 — Generate review")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        generate_btn = st.button(
            "Generate Literature Review",
            type="primary",
            disabled=not uploaded_files or len(uploaded_files) < 2,
            use_container_width=True,
            help="Minimum 2 papers required"
        )

    with col2:
        if st.session_state.review_data:
            clear_btn = st.button(
                "Clear Review",
                use_container_width=True
            )
            if clear_btn:
                st.session_state.review_data   = None
                st.session_state.review_papers = []
                st.rerun()

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 PDF papers to generate a review.")

    if generate_btn and uploaded_files and len(uploaded_files) >= 2:
        _run_review_pipeline(uploaded_files, topic)

    if st.session_state.review_data:
        _render_review_output(st.session_state.review_data)
    
    
    # ── Compare Papers Tab ────────────────────────────────────────────────────────
def render_compare_papers_tab():
    st.markdown(
        '<p class="main-header">🔬 Compare Papers</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload 2–5 research papers and get a '
        'complete side-by-side comparison — scored table across 7 dimensions '
        'plus deep narrative analysis. Exports to Word and PDF.</p>',
        unsafe_allow_html=True
    )

    # ── Step 1: Upload ────────────────────────────────────────
    st.markdown("### Step 1 — Upload papers to compare")
    st.caption(
        "Upload 2–5 papers. Works best when papers are on the "
        "same or related topics."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="compare_uploader",
        help="Upload 2–5 research papers to compare"
    )

    if uploaded_files:
        n = len(uploaded_files)
        if n < 2:
            st.warning("Upload at least 2 papers to compare.")
        elif n > 5:
            st.warning("Maximum 5 papers supported. First 5 will be used.")
            uploaded_files = uploaded_files[:5]
        else:
            st.success(f"✅ {n} papers selected:")

        for f in uploaded_files:
            size_kb = round(f.size / 1024, 1)
            st.markdown(f"  - `{f.name}` ({size_kb} KB)")

    st.markdown("---")

    # ── Step 2: Options ───────────────────────────────────────
    st.markdown("### Step 2 — Comparison options")

    col1, col2 = st.columns(2)
    with col1:
        show_scores = st.checkbox(
            "Part A — Score table",
            value=True,
            help="Side-by-side scores per dimension"
        )
    with col2:
        show_narrative = st.checkbox(
            "Part B — Deep narrative",
            value=True,
            help="Detailed paragraph analysis per dimension"
        )

    st.markdown("---")

    # ── Step 3: Compare button ────────────────────────────────
    st.markdown("### Step 3 — Run comparison")

    col1, col2 = st.columns([2, 1])

    with col1:
        compare_btn = st.button(
            "Compare Papers",
            type="primary",
            disabled=not uploaded_files or len(uploaded_files) < 2,
            use_container_width=True,
            help="Minimum 2 papers required"
        )

    with col2:
        if st.session_state.comparison_data:
            if st.button("Clear Results", use_container_width=True):
                st.session_state.comparison_data   = None
                st.session_state.comparison_papers = []
                st.rerun()

    if not uploaded_files or len(uploaded_files) < 2:
        st.info("Upload at least 2 PDF papers to start comparing.")

    # ── Run pipeline ──────────────────────────────────────────
    if compare_btn and uploaded_files and len(uploaded_files) >= 2:
        _run_comparison_pipeline(uploaded_files)

    # ── Show results ──────────────────────────────────────────
    if st.session_state.comparison_data:
        _render_comparison_output(
            st.session_state.comparison_data,
            show_scores=show_scores,
            show_narrative=show_narrative
        )


def _run_comparison_pipeline(uploaded_files):
    """Runs the full paper comparison pipeline with live progress bar."""
    st.markdown("---")
    st.markdown("### Comparing papers...")

    progress_bar    = st.progress(0)
    status_text     = st.empty()
    error_container = st.empty()

    try:
        # Phase A: Save files
        status_text.text("Saving uploaded files...")
        progress_bar.progress(5)

        temp_paths = []
        for f in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(f.getvalue())
                temp_paths.append(tmp.name)

        # Phase B: Extract metadata
        status_text.text(
            f"Extracting metadata from {len(uploaded_files)} papers..."
        )
        progress_bar.progress(15)

        metadata_list = process_multiple_papers(temp_paths)

        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

        if not metadata_list or len(metadata_list) < 2:
            error_container.error(
                "❌ Could not extract metadata from enough papers. "
                "Please check your PDFs are text-based."
            )
            return

        progress_bar.progress(25)
        status_text.text(
            f"Metadata extracted from {len(metadata_list)} papers. "
            "Running comparison analysis..."
        )

        # Phase C: Generate comparison
        total_steps = 9  # 1 scoring + 7 narrative + 1 done

        def progress_callback(step: int, total: int, message: str):
            pct = 25 + int((step / total) * 70)
            progress_bar.progress(pct)
            status_text.text(f"[{step}/{total}] {message}")

        comparison_data = generate_full_comparison(
            metadata_list=metadata_list,
            progress_callback=progress_callback
        )

        progress_bar.progress(100)
        status_text.text("Comparison complete!")

        st.session_state.comparison_data   = comparison_data
        st.session_state.comparison_papers = metadata_list

        winner = comparison_data.get("winner", "Unknown")
        st.success(
            f"✅ Comparison complete!  \n"
            f"Compared **{len(metadata_list)} papers** across "
            f"**7 dimensions**.  \n"
            f"🏆 Winner: **{winner}**"
        )
        st.rerun()

    except Exception as e:
        error_container.error(f"❌ Error: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def _render_comparison_output(
    comparison_data: dict,
    show_scores: bool = True,
    show_narrative: bool = True
):
    """Renders the comparison results with download buttons."""
    import pandas as pd

    papers     = comparison_data["papers"]
    winner     = comparison_data["winner"]
    winner_idx = comparison_data["winner_index"]
    totals     = comparison_data["totals"]
    narratives = comparison_data.get("narratives", {})

    st.markdown("---")

    # ── Winner banner ─────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #28a745, #20c997);
            padding: 1rem 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
        ">
            <div style="color:white;font-size:1.4rem;font-weight:700;">
                🏆 Overall Winner
            </div>
            <div style="color:#d4edda;font-size:1.1rem;margin-top:0.3rem;">
                {winner}
            </div>
            <div style="color:#a8d5b5;font-size:0.85rem;margin-top:0.3rem;">
                Score: {totals[winner_idx]}/{(len(comparison_data['scores']))*10}
                &nbsp;|&nbsp;
                {comparison_data.get('winner_reason', '')}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ── Download buttons ──────────────────────────────────────
    st.markdown("### Download")
    col1, col2 = st.columns(2)

    with col1:
        try:
            docx_bytes = export_comparison_to_docx(comparison_data)
            fname      = generate_comparison_filename(papers, "docx")
            st.download_button(
                label="Download Word (.docx)",
                data=docx_bytes,
                file_name=fname,
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"DOCX export failed: {str(e)}")

    with col2:
        try:
            pdf_bytes = export_comparison_to_pdf(comparison_data)
            fname     = generate_comparison_filename(papers, "pdf")
            st.download_button(
                label="Download PDF (.pdf)",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")

    st.markdown("---")

    # ── Part A: Score table ───────────────────────────────────
    if show_scores:
        st.markdown("## Part A — Comparative Score Table")
        st.caption(
            "🟢 Strong (8–10)  ·  🟡 Moderate (6–7)  ·  🔴 Weak (1–5)"
        )

        rows = comparison_to_table_rows(comparison_data)
        df   = pd.DataFrame(rows)

        def color_score(val):
            """Apply background color to score cells."""
            try:
                v = int(str(val).replace("🏆", "").strip())
                if v >= 8:
                    return "background-color: #d4edda; color: #155724;"
                elif v >= 6:
                    return "background-color: #fff3cd; color: #856404;"
                elif v > 0:
                    return "background-color: #f8d7da; color: #721c24;"
            except Exception:
                pass
            return ""

        styled = df.style.applymap(
            color_score,
            subset=[c for c in df.columns if c != "Dimension"]
        )

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True
        )

        # Per-paper score summary
        st.markdown("#### Score summary")
        summary_cols = st.columns(len(papers))
        for i, (col, paper) in enumerate(zip(summary_cols, papers)):
            with col:
                short = paper[:30] + "..." if len(paper) > 30 else paper
                is_winner = (i == winner_idx)
                border = "2px solid #28a745" if is_winner else "1px solid #ddd"
                bg     = "#f0fff4" if is_winner else "#f9f9f9"
                trophy = " 🏆" if is_winner else ""
                st.markdown(
                    f"""<div style="
                        border: {border};
                        border-radius: 8px;
                        padding: 0.8rem;
                        text-align: center;
                        background: {bg};
                        color: #1a1a1a;
                    ">
                        <div style="font-weight:700;font-size:0.85rem;
                                    color:#1a1a1a;">{short}{trophy}</div>
                        <div style="font-size:1.4rem;font-weight:700;
                                    color:{'#28a745' if is_winner else '#1f77b4'};
                                    margin-top:0.3rem;">
                            {totals[i]}
                        </div>
                        <div style="font-size:0.75rem;color:#666;">
                            / {len(comparison_data['scores'])*10} pts
                        </div>
                    </div>""",
                    unsafe_allow_html=True
                )

        st.markdown("---")

    # ── Part B: Narrative ─────────────────────────────────────
    if show_narrative:
        st.markdown("## Part B — Deep Narrative Analysis")

        from utils.comparison_chain import DIMENSION_LABELS

        for i, (dim, label) in enumerate(DIMENSION_LABELS.items(), 1):
            icon = {
                "research_problem": "🎯",
                "methodology":      "⚙️",
                "datasets":         "📊",
                "results":          "📈",
                "contributions":    "💡",
                "limitations":      "⚠️",
                "overall":          "🏆"
            }.get(dim, "📝")

            with st.expander(
                f"{icon} {i}. {label}",
                expanded=(dim == "overall")
            ):
                text = narratives.get(dim, "Analysis not available.")
                st.markdown(
                    f'<div class="answer-box">{text}</div>',
                    unsafe_allow_html=True
                )

    # ── Papers info ───────────────────────────────────────────
    with st.expander(
        f"Papers compared ({comparison_data['paper_count']})",
        expanded=False
    ):
        for i, paper in enumerate(
            comparison_data.get("metadata", []), 1
        ):
            is_winner = (i - 1) == winner_idx
            trophy    = " 🏆" if is_winner else ""
            st.markdown(
                f"**{i}. {paper.get('title', 'Unknown')}{trophy}**  \n"
                f"Authors: {', '.join(paper.get('authors', ['Unknown']))}  \n"
                f"Year: {paper.get('year', 'Unknown')} · "
                f"Domain: {paper.get('domain', 'Unknown')}  \n"
                f"Keywords: {', '.join(paper.get('keywords', []))}"
            )
            st.markdown("---")


def _run_review_pipeline(uploaded_files, topic: str):
    """Runs the full literature review generation pipeline."""
    st.markdown("---")
    st.markdown("### Generating your literature review...")

    progress_bar    = st.progress(0)
    status_text     = st.empty()
    error_container = st.empty()

    try:
        status_text.text("Saving uploaded files...")
        progress_bar.progress(5)

        temp_paths = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".pdf"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_paths.append(tmp.name)

        status_text.text(
            f"Extracting metadata from {len(uploaded_files)} papers..."
        )
        progress_bar.progress(15)

        metadata_list = process_multiple_papers(temp_paths)

        for path in temp_paths:
            try:
                os.unlink(path)
            except Exception:
                pass

        if not metadata_list:
            error_container.error(
                "❌ Could not extract metadata from any paper. "
                "Please check your PDFs."
            )
            return

        progress_bar.progress(30)
        status_text.text(
            f"Metadata extracted from {len(metadata_list)} papers. "
            "Generating review sections..."
        )

        def progress_callback(step: int, total: int, message: str):
            pct = 30 + int((step / total) * 60)
            progress_bar.progress(pct)
            status_text.text(f"[{step}/{total}] {message}")

        review_data = generate_full_review(
            metadata_list=metadata_list,
            topic=topic.strip() if topic.strip() else "",
            progress_callback=progress_callback
        )

        progress_bar.progress(100)
        status_text.text("Review complete!")

        st.session_state.review_data   = review_data
        st.session_state.review_papers = metadata_list
        st.session_state.review_topic  = review_data.get("topic", "")

        st.success(
            f"✅ Literature review generated successfully!  \n"
            f"Covered **{len(metadata_list)} papers** across **6 sections**."
        )
        st.rerun()

    except Exception as e:
        error_container.error(f"❌ Error generating review: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def _render_review_output(review_data: dict):
    """Renders the generated review with download buttons."""
    import pandas as pd

    st.markdown("---")
    st.markdown(
        f"## Review: {review_data.get('topic', 'Research Review')}"
    )
    st.caption(
        f"Based on {review_data.get('paper_count', 0)} papers · "
        f"Generated {datetime.now().strftime('%B %d, %Y')}"
    )

    # ── Download buttons ──────────────────────────────────────
    st.markdown("### Download")
    col1, col2 = st.columns(2)

    with col1:
        try:
            docx_bytes = export_to_docx(review_data)
            fname      = generate_filename(
                review_data.get("topic", "review"), "docx"
            )
            st.download_button(
                label="Download Word (.docx)",
                data=docx_bytes,
                file_name=fname,
                mime=(
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
                use_container_width=True
            )
        except Exception as e:
            st.error(f"DOCX export failed: {str(e)}")

    with col2:
        try:
            pdf_bytes = export_to_pdf(review_data)
            fname     = generate_filename(
                review_data.get("topic", "review"), "pdf"
            )
            st.download_button(
                label="Download PDF (.pdf)",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")

    st.markdown("---")

    with st.expander("1. Introduction", expanded=True):
        st.markdown(
            f'<div class="answer-box">'
            f'{review_data.get("introduction", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander("2. Related Work", expanded=False):
        st.markdown(
            f'<div class="answer-box">'
            f'{review_data.get("related_work", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander("3. Comparison Table", expanded=True):
        comparison_data = review_data.get("comparison_data", [])
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Title":       st.column_config.TextColumn(
                        "Title",       width="medium"),
                    "Authors":     st.column_config.TextColumn(
                        "Authors",     width="small"),
                    "Year":        st.column_config.TextColumn(
                        "Year",        width="small"),
                    "Method":      st.column_config.TextColumn(
                        "Method",      width="medium"),
                    "Dataset":     st.column_config.TextColumn(
                        "Dataset",     width="small"),
                    "Key Result":  st.column_config.TextColumn(
                        "Key Result",  width="large"),
                    "Limitations": st.column_config.TextColumn(
                        "Limitations", width="medium"),
                }
            )
        else:
            st.info("No comparison data available.")

    with st.expander("4. Research Gaps", expanded=False):
        st.markdown(
            f'<div class="answer-box">'
            f'{review_data.get("gaps", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander("5. Future Research Directions", expanded=False):
        st.markdown(
            f'<div class="answer-box">'
            f'{review_data.get("future_directions", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander("6. Conclusion", expanded=False):
        st.markdown(
            f'<div class="answer-box">'
            f'{review_data.get("conclusion", "")}'
            f'</div>',
            unsafe_allow_html=True
        )

    with st.expander(
        f"Papers included ({review_data.get('paper_count', 0)})",
        expanded=False
    ):
        for i, paper in enumerate(review_data.get("papers", []), 1):
            st.markdown(
                f"**{i}. {paper.get('title', 'Unknown')}**  \n"
                f"Authors: {', '.join(paper.get('authors', ['Unknown']))}  \n"
                f"Year: {paper.get('year', 'Unknown')} · "
                f"Domain: {paper.get('domain', 'Unknown')}  \n"
                f"Keywords: {', '.join(paper.get('keywords', []))}"
            )
            st.markdown("---")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    init_session_state()

    tab1, tab2, tab3 = st.tabs([
        "💬 Research Chat",
        "📝 Literature Review",
        "🔬 Compare Papers"
    ])

    with tab1:
        render_sidebar()
        render_chat()

    with tab2:
        render_literature_review_tab()

    with tab3:
        render_compare_papers_tab()


if __name__ == "__main__":
    main()