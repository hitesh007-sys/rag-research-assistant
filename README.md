# 📄 RAG Research Assistant

A production-grade **Multi-Paper AI Research System** that combines Retrieval-Augmented Generation with intelligent research tools. Upload PDFs, ask questions, generate literature reviews, and compare papers — all grounded in your documents with zero hallucination.

Built with Python, Streamlit, LangChain, ChromaDB, Groq Llama 3, and Sentence Transformers.

---

## 🌐 Live Demo

👉 [Try it live](https://hitesh-rag-assistant.streamlit.app)

---

## ✨ Features

### 💬 Research Chat
- **PDF Upload** — Upload single or multiple PDF documents
- **Semantic Search** — Find relevant chunks using vector embeddings
- **Hybrid Search** — BM25 keyword + semantic search combined (60/40 weighted)
- **Reranking** — Cross-encoder reranks chunks for higher accuracy
- **Streaming Responses** — Answers stream word by word like ChatGPT
- **Conversation Memory** — Remembers last 5 exchanges for natural follow-ups
- **Question Condensing** — Vague follow-ups rewritten as full standalone questions
- **Source Citations** — Every answer shows which document chunks were used
- **Upload History** — Tracks all previously processed PDFs with metadata
- **Multi-PDF Support** — Query across multiple documents simultaneously

### 📝 Auto Literature Review
- **Metadata Extraction** — Extracts title, authors, year, abstract, keywords, methodology, datasets, findings and limitations from each paper
- **6 Auto-generated Sections** — Introduction, Related Work, Comparison Table, Research Gaps, Future Directions, Conclusion
- **Export to Word** — Fully formatted .docx with blue headings, colour-coded table and references
- **Export to PDF** — Clean A4 PDF with cover page, table and all sections
- **Progress tracking** — Live progress bar as each section generates

### 🔬 Compare Papers
- **7 Comparison Dimensions** — Research Problem, Methodology, Datasets, Results, Contributions, Limitations, Overall Verdict
- **Part A: Score Table** — Papers scored 1–10 per dimension with colour coding (🟢 8–10, 🟡 6–7, 🔴 1–5)
- **Part B: Deep Narrative** — Paragraph-by-paragraph analysis per dimension
- **Winner Detection** — Automatically identifies the strongest paper with reasoning
- **Export to Word and PDF** — Full comparison report with colour-coded tables

---

## 🏗️ Project Structure

```
rag-research-assistant/
├── app.py                        # Main Streamlit UI — 3 tabs
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # One-command launcher
├── packages.txt                  # System packages for cloud
├── runtime.txt                   # Python version for cloud
├── .env                          # API keys (never commit)
├── .env.example                  # Safe template to share
├── .gitignore
├── .streamlit/
│   └── config.toml               # Streamlit configuration
├── data/
│   ├── chroma_db/                # Persisted vector store
│   └── upload_history.json       # Document history
└── utils/
    ├── __init__.py
    ├── pdf_loader.py             # PDF extraction + chunking
    ├── embeddings.py             # ChromaDB vector operations
    ├── qa_chain.py               # LLM chain + memory + streaming
    ├── hybrid_retriever.py       # BM25 + semantic hybrid search
    ├── reranker.py               # Cross-encoder reranking
    ├── history_manager.py        # Upload history tracker
    ├── paper_metadata.py         # Research paper metadata extractor
    ├── review_chain.py           # Literature review section generator
    ├── report_exporter.py        # DOCX + PDF export for reviews
    ├── comparison_chain.py       # Paper comparison scoring + narrative
    └── comparison_exporter.py    # DOCX + PDF export for comparisons
```

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq Llama 3.3-70b-versatile |
| Orchestration | LangChain |
| Vector Store | ChromaDB (local persistent) |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF Parsing | PyPDF + pdfplumber |
| Hybrid Search | BM25 (rank-bm25) + Semantic |
| Document Export | python-docx + ReportLab |
| Data Processing | Pandas |
| Containerisation | Docker + Docker Compose |
| Deployment |  Streamlit Cloud |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Groq API key — free at [console.groq.com](https://console.groq.com)

### Option A — Local Python

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/hitesh007-sys/rag-research-assistant
cd rag-research-assistant
```

**Step 2 — Create virtual environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate.bat

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

**Step 5 — Run the app:**
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

### Option B — Docker (Recommended)

Run the entire app in an isolated container with a single command — no Python installation required.

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)

#### Step 1 — Clone the repository:
```bash
git clone https://github.com/hitesh007-sys/rag-research-assistant
cd rag-research-assistant
```

#### Step 2 — Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

#### Step 3 — Start with Docker Compose:
```bash
docker-compose up --build
```

Open `http://localhost:8501` in your browser.

> The `--build` flag is only needed the first time (or after changing dependencies). For subsequent starts, use `docker-compose up`.

#### Useful Docker Commands

```bash
# Run in detached (background) mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down

# Rebuild after dependency changes
docker-compose up --build

# Remove containers and volumes (wipes ChromaDB data)
docker-compose down -v
```

#### Data Persistence

The `data/` directory is mounted as a volume so your ChromaDB vector store and upload history survive container restarts:

```yaml
volumes:
  - ./data:/app/data
```

To reset all stored data, delete the `data/chroma_db/` folder or run `docker-compose down -v`.

---

## 📖 How to Use

### 💬 Research Chat Tab
1. Upload a PDF using the sidebar → click **"Process PDF"**
2. Add more PDFs → click **"Add to existing"** for multi-document queries
3. Type your question in the chat box
4. Click **"View source chunks"** to see citations
5. Ask follow-up questions naturally — memory handles context

### 📝 Literature Review Tab
1. Upload 2–10 research PDFs
2. Optionally type your topic (e.g. `"Transformer models in NLP"`)
3. Click **"Generate Literature Review"**
4. Watch the live progress bar through all 6 sections
5. Download as Word or PDF

### 🔬 Compare Papers Tab
1. Upload 2–5 research PDFs
2. Select Part A (scores) and/or Part B (narrative)
3. Click **"Compare Papers"**
4. See the winner banner, colour-coded score table and narrative analysis
5. Download the full comparison report as Word or PDF

---

## ⚙️ Configuration

Edit `.env` to configure:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_K=4
```

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `GROQ_MODEL_NAME` | llama-3.3-70b-versatile | Groq model to use |
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `RETRIEVAL_K` | 4 | Chunks retrieved per query |

---

## 🔄 RAG Pipeline

```
PDF Upload → Text Extraction → Chunking → Embeddings
                                               ↓
User Query → Query Embedding → Hybrid Search (BM25 + Semantic)
                                               ↓
                              Reranking (Cross-encoder)
                                               ↓
                              Top-k Chunks → LLM Prompt
                                               ↓
                    Memory-aware Question Condensing
                                               ↓
                         Streaming Answer + Citations
```

---

## 📊 Literature Review Pipeline

```
Upload PDFs → Extract Metadata (title, authors, year, abstract,
              methodology, datasets, findings, limitations)
                                               ↓
              Generate 6 sections via targeted LLM prompts:
              Introduction → Related Work → Comparison Table
              → Research Gaps → Future Directions → Conclusion
                                               ↓
              Export to .docx (editable) + .pdf (shareable)
```

---

## 🔬 Paper Comparison Pipeline

```
Upload 2-5 PDFs → Extract Metadata (reuses literature review extractor)
                                               ↓
              Score each paper (1-10) across 6 dimensions in one LLM call
              + Generate deep narrative per dimension (7 LLM calls)
                                               ↓
              Part A: Colour-coded score table + winner detection
              Part B: Paragraph analysis per dimension
                                               ↓
              Export to .docx + .pdf
```

---

## 🛠️ Available Groq Models

| Model | Speed | Quality | Best for |
|---|---|---|---|
| `llama-3.3-70b-versatile` | Medium | Best | Research tasks |
| `llama-3.1-8b-instant` | Fastest | Good | Quick queries |
| `mixtral-8x7b-32768` | Fast | Very good | Long context |

---

## 🗺️ Roadmap

- [x] PDF upload and RAG chat
- [x] Hybrid search (BM25 + Semantic)
- [x] Cross-encoder reranking
- [x] Conversation memory with question condensing
- [x] Multi-PDF support
- [x] Upload history tracking
- [x] Auto literature review generator
- [x] Paper comparison engine
- [x] Docker + Docker Compose support
- [ ] Citation graph visualization
- [ ] Hypothesis generator
- [ ] ArXiv paper search integration
- [ ] Research timeline view

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration
- [Groq](https://groq.com) — Fast LLM inference
- [ChromaDB](https://trychroma.com) — Vector database
- [Streamlit](https://streamlit.io) — Web UI framework
- [Sentence Transformers](https://sbert.net) — Embeddings + reranking
- [python-docx](https://python-docx.readthedocs.io) — Word document export
- [ReportLab](https://www.reportlab.com) — PDF export

---

## 📝 License

MIT License — feel free to use, modify and build upon this project.

---

## 👨‍💻 Author

**Hitesh Kumar Sahu**
- GitHub: [@hitesh007-sys](https://github.com/hitesh007-sys)
- Project: [rag-research-assistant](https://github.com/hitesh007-sys/rag-research-assistant)