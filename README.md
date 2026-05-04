# 📄 RAG Research Assistant

A production-grade **Retrieval-Augmented Generation (RAG)** application 
that lets you upload PDF documents and ask questions in natural language. 
Answers are grounded in your documents with source citations.

Built with Python, Streamlit, LangChain, ChromaDB, Groq Llama 3, and 
Sentence Transformers.

---

## 🚀 Key Features

- 📂 **Multi-PDF Upload**  
  Query across multiple documents at once  

- 🔍 **Hybrid Retrieval**  
  Combines BM25 keyword search + semantic search  

- 🎯 **Reranking**  
  Cross-encoder improves answer relevance  

- 💬 **Conversational Memory**  
  Handles follow-up questions intelligently  

- ⚡ **Streaming Responses**  
  Real-time answer generation like ChatGPT  

- 📑 **Source Citations**  
  Transparent answers with document references  

- 🧠 **Question Rewriting**  
  Converts vague queries into structured questions  

- 🗂️ **Upload History Tracking**  
  Maintains processed document records  

---

## 🏗️ Project Structure

rag-research-assistant/
│
├── app.py                  # Streamlit entry point  
├── requirements.txt        # Dependencies  
├── packages.txt            # System dependencies  
├── runtime.txt             # Python version  
├── .env                    # API keys (not committed)  
├── .gitignore  
│
├── data/
│   └── upload_history.json
│
├── utils/
│   ├── embeddings.py
│   ├── hybrid_retriever.py
│   ├── reranker.py
│   ├── qa_chain.py
│   ├── review_chain.py
│   ├── history_manager.py
│   ├── pdf_loader.py
│   └── report_exporter.py
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

## 🌍 Deployment (Streamlit)

This app can be deployed on:

- Streamlit Community Cloud  
- Render  
- Railway  

### Steps (Streamlit Cloud)

1. Push code to GitHub  
2. Open Streamlit Cloud  
3. Select your repository  
4. Set main file → `app.py`  
5. Add environment variables  

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

1. 📄 PDF Upload  
2. ✂️ Text Extraction  
3. 🔹 Chunking  
4. 🧠 Embeddings  
5. ❓ User Query  
6. 🔍 Hybrid Retrieval (BM25 + Semantic)  
7. 🎯 Reranking (Cross-Encoder)  
8. 🤖 LLM Generation (Groq)  
9. ⚡ Streaming Response + Citations  
---

## 🚀 Why This Project Stands Out

- Combines multiple retrieval strategies (not just basic RAG)  
- Implements reranking for better accuracy  
- Supports multi-document reasoning  
- Includes literature review generation pipeline  
- Designed for real-world research workflows  

---

## 🛠️ Available Models

| Model | Speed | Quality |
|---|---|---|
| llama-3.3-70b-versatile | Medium | Best |
| llama-3.1-8b-instant | Fastest | Good |
| mixtral-8x7b-32768 | Fast | Very good |

---

## 🔮 Future Improvements

- 📈 Citation graph visualization  
- 📄 Export to PDF / Word reports  
- 🌍 Multi-language support  
- 🧠 Fine-tuned domain-specific models  

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