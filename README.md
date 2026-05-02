# 📄 RAG Research Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** application 
that lets you upload PDF documents and ask questions in natural language. 
Answers are grounded in your documents with source citations.

Built with Python, Streamlit, LangChain, ChromaDB, Groq Llama 3, and 
Sentence Transformers.

---

## 🎯 Features

- **PDF Upload** — Upload single or multiple PDF documents
- **Semantic Search** — Find relevant chunks using vector embeddings
- **Hybrid Search** — BM25 keyword + semantic search combined
- **Reranking** — Cross-encoder reranks chunks for higher accuracy
- **Streaming Responses** — Answers stream word by word like ChatGPT
- **Conversation Memory** — Remembers last 5 exchanges for follow-ups
- **Question Condensing** — Vague follow-ups rewritten as full questions
- **Source Citations** — Every answer shows which document chunks were used
- **Upload History** — Tracks all previously processed PDFs
- **Multi-PDF Support** — Query across multiple documents simultaneously
- **Docker Deployment** — One command to run anywhere

---

## 🏗️ Project Structure
rag-research-assistant/
├── app.py                    # Main Streamlit UI
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # One-command launcher
├── .env                      # API keys (never commit)
├── .gitignore
├── data/
│   ├── chroma_db/            # Persisted vector store
│   └── upload_history.json   # Document history
└── utils/
├── init.py
├── pdf_loader.py         # PDF extraction + chunking
├── embeddings.py         # ChromaDB vector operations
├── qa_chain.py           # LLM chain + memory + streaming
├── hybrid_retriever.py   # BM25 + semantic hybrid search
├── reranker.py           # Cross-encoder reranking
└── history_manager.py    # Upload history tracker
---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq Llama 3.3-70b |
| Orchestration | LangChain |
| Vector Store | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| PDF Parsing | PyPDF |
| Hybrid Search | BM25 + Semantic |
| Deployment | Docker |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/yourusername/rag-research-assistant
cd rag-research-assistant
```

**Step 2 — Create virtual environment:**
```bash
python -m venv venv

# Windows
venv\Scripts\activate.bat

# Mac/Linux
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

## 🐳 Docker Deployment

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# Stop
docker compose down
```

---

## 📖 How to Use

1. **Upload a PDF** — Click "Browse files" in the sidebar
2. **Process PDF** — Click "Process PDF" to extract and embed
3. **Add more PDFs** — Click "Add to existing" for multi-PDF queries
4. **Ask questions** — Type in the chat box at the bottom
5. **View sources** — Click "View source chunks" to see citations
6. **Follow-ups** — Ask follow-up questions naturally using memory

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
| GROQ_API_KEY | required | Your Groq API key |
| GROQ_MODEL_NAME | llama-3.3-70b-versatile | Groq model to use |
| CHUNK_SIZE | 500 | Characters per chunk |
| CHUNK_OVERLAP | 50 | Overlap between chunks |
| RETRIEVAL_K | 4 | Chunks retrieved per query |

---

## 🔄 RAG Pipeline
PDF Upload → Text Extraction → Chunking → Embeddings
↓
User Query → Query Embedding → Hybrid Search (BM25 + Semantic)
↓
Reranking (Cross-encoder)
↓
Top-k Chunks → LLM Prompt
↓
Streaming Answer + Citations
---

## 🛠️ Available Models

| Model | Speed | Quality |
|---|---|---|
| llama-3.3-70b-versatile | Medium | Best |
| llama-3.1-8b-instant | Fastest | Good |
| mixtral-8x7b-32768 | Fast | Very good |

---

## 📝 License

MIT License — feel free to use and modify.

---

## 🙏 Acknowledgements

- [LangChain](https://langchain.com) — LLM orchestration
- [Groq](https://groq.com) — Fast LLM inference
- [ChromaDB](https://trychroma.com) — Vector database
- [Streamlit](https://streamlit.io) — Web UI framework
- [Sentence Transformers](https://sbert.net) — Embeddings + reranking