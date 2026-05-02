# utils/qa_chain.py
import os
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
RETRIEVAL_K     = int(os.getenv("RETRIEVAL_K", 4))

# NOTE: reranker and hybrid_retriever are imported LAZILY inside functions
# so they never crash on platforms where torch is unavailable (Streamlit Cloud)


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm(streaming: bool = False) -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not found. Please add it to your .env file."
        )
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL_NAME,
        temperature=0.2,
        max_tokens=1024,
        streaming=streaming
    )


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        template="""You are a helpful research assistant.
Use ONLY the following context to answer the question.
If the answer is not in the context, say "I don't have enough
information in the provided documents to answer this question."
Always be concise and accurate.

Context:
{context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )


# ── Memory ────────────────────────────────────────────────────────────────────
def build_memory(k: int = 5) -> ConversationBufferWindowMemory:
    return ConversationBufferWindowMemory(
        k=k,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True
    )


# ── Basic QA chain (no memory) ────────────────────────────────────────────────
def build_qa_chain(vector_store) -> RetrievalQA:
    llm       = get_llm()
    prompt    = build_prompt_template()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print(f"QA chain ready — model: {GROQ_MODEL_NAME}, "
          f"top {RETRIEVAL_K} chunks")
    return qa_chain


def get_answer(qa_chain, question: str) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    result = qa_chain.invoke({"query": question})
    return {
        "answer":           result["result"],
        "source_documents": result["source_documents"]
    }


# ── Conversational chain (memory + hybrid retriever) ─────────────────────────
def build_conversational_chain(
    vector_store,
    use_hybrid: bool = True
) -> ConversationalRetrievalChain:
    """
    Builds ConversationalRetrievalChain with memory.
    Hybrid retriever is imported lazily — safe on all platforms.
    """
    llm    = get_llm()
    memory = build_memory(k=5)

    # Lazy import of hybrid retriever
    if use_hybrid:
        try:
            from utils.hybrid_retriever import build_hybrid_retriever
            retriever = build_hybrid_retriever(vector_store)
            print("Using hybrid retriever (semantic + BM25)")
        except Exception as e:
            print(f"Hybrid retriever failed ({e}) — using semantic only")
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RETRIEVAL_K}
            )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K}
        )

    answer_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful research assistant with memory of
our conversation. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough
information in the provided documents to answer this question."
Be concise, accurate, and refer to previous answers when relevant.

Context:
{context}

Question: {question}

Answer:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": answer_prompt},
        verbose=False
    )

    print("Conversational chain ready.")
    return chain


def get_conversational_answer(chain, question: str) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    result = chain.invoke({"question": question})
    return {
        "answer":           result["answer"],
        "source_documents": result["source_documents"]
    }


# ── Streaming answer with memory-aware condensing ────────────────────────────
def get_streaming_answer(chain, question: str):
    """
    Streams answer token by token with:
    - Question condensing using chat history
    - Hybrid retrieval for relevant chunks
    - Optional reranking (lazy import — safe on cloud)
    - Memory saved after each response
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    # Step 1 — Get memory history
    memory_vars  = chain.memory.load_memory_variables({})
    chat_history = memory_vars.get("chat_history", [])

    history_text = ""
    if chat_history:
        for msg in chat_history:
            if hasattr(msg, "type"):
                prefix = "Human" if msg.type == "human" else "Assistant"
                history_text += f"{prefix}: {msg.content}\n"

    # Step 2 — Condense follow-up into standalone question
    condense_llm = get_llm(streaming=False)

    if history_text:
        condense_prompt = f"""Given this conversation history and a follow-up
question, rewrite the follow-up as a complete standalone question.
If it's already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Follow-up question: {question}

Standalone question:"""

        condensed       = condense_llm.invoke(condense_prompt)
        search_question = condensed.content.strip()
        print(f"Original: '{question}' → Condensed: '{search_question}'")
    else:
        search_question = question

    # Step 3 — Retrieve docs
    retriever = chain.retriever
    try:
        docs = retriever.invoke(search_question)
    except Exception:
        try:
            docs = retriever.vectorstore.similarity_search(
                search_question, k=RETRIEVAL_K
            )
        except Exception:
            docs = []

    # Step 3b — Rerank if enabled (lazy import — never crashes on cloud)
    use_reranking = os.environ.get("USE_RERANKING", "true").lower() == "true"
    if use_reranking and docs:
        try:
            from utils.reranker import rerank_documents, is_reranker_available
            if is_reranker_available():
                docs = rerank_documents(
                    query=search_question,
                    documents=docs,
                    top_k=RETRIEVAL_K
                )
        except Exception as e:
            print(f"Reranking skipped: {e}")

    context = "\n\n".join([doc.page_content for doc in docs])

    # Step 4 — Stream final answer
    stream_llm = get_llm(streaming=True)

    final_prompt = f"""You are a helpful research assistant with memory
of our conversation. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough
information in the provided documents to answer this question."
Be detailed and thorough in your explanation.

Previous conversation:
{history_text if history_text else "No previous conversation."}

Context from document:
{context}

Question: {question}

Answer:"""

    full_answer = ""
    for chunk in stream_llm.stream(final_prompt):
        if chunk.content:
            full_answer += chunk.content
            yield chunk.content

    # Step 5 — Save to memory
    chain.memory.save_context(
        {"question": question},
        {"answer": full_answer}
    )