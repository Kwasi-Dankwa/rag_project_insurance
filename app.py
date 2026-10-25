import json
from pathlib import Path

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI


# Keep all application files relative to the folder containing app.py
BASE_DIR = Path(__file__).resolve().parent

CONFIG_FILE_PATH = BASE_DIR / "config.json"
FAISS_INDEX_PATH = BASE_DIR / "faiss_index"

MODEL = "gpt-4o-mini"
EMBED_MODEL = "surajvbangera/mediclaim_embedding"
TOP_K = 3


def load_config():
    """Load API configuration from config.json."""
    if not CONFIG_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE_PATH}"
        )

    with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    openai_api_key = config.get("OPENAI_API_KEY")
    openai_api_base = config.get("OPENAI_API_BASE")

    if not openai_api_key or not openai_api_base:
        raise ValueError(
            "config.json must contain OPENAI_API_KEY and OPENAI_API_BASE."
        )

    return openai_api_key, openai_api_base


@st.cache_resource
def load_resources():
    """Load the embedding model, FAISS index, and generator LLM once."""
    openai_api_key, openai_api_base = load_config()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.load_local(
        str(FAISS_INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
        top_p=1,
        max_tokens=512,
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    return vectorstore, llm


def build_context(docs):
    """Format retrieved chunks with source metadata for the LLM."""
    blocks = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get(
            "source_file",
            Path(doc.metadata.get("source", "Unknown source")).name,
        )
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")

        blocks.append(
            f"[Source {i} | {source} | {chunk_id}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(blocks)


def answer_query(query, vectorstore, llm):
    """Retrieve relevant policy chunks and generate a grounded answer."""
    retrieved_docs = vectorstore.similarity_search(query, k=TOP_K)

    context = build_context(retrieved_docs)

    prompt = f"""
You are an AI assistant helping MeridianHealth insurance representatives
answer policy-related questions.

User Query:
{query}

Retrieved Policy Context:
{context}

Instructions:
1. Answer using only the retrieved policy context.
2. Do not use outside knowledge or invent policy details.
3. Include important conditions, exclusions, limitations, and waiting periods
   when relevant.
4. Answer every part of a multi-part question.
5. Keep the response concise and clear.
6. If the retrieved context is insufficient, say:
   "The available policy documents do not contain enough information to answer this question."
7. If the retrieved context contains conflicting information, clearly flag it.

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, retrieved_docs


st.set_page_config(
    page_title="MeridianHealth RAG Assistant",
    page_icon="🩺",
    layout="wide",
)

st.title("MeridianHealth RAG Assistant")
st.caption("Context-aware answers from MeridianHealth policy documents")


try:
    vectorstore, llm = load_resources()
except Exception as exc:
    st.error(
        "Could not load the RAG system. Make sure `config.json` and the "
        "`faiss_index` folder are present in the same project folder."
    )
    st.caption(f"Details: {exc}")
    st.stop()


query = st.chat_input("Ask a policy question...")

if query:
    with st.spinner("Searching policy documents and generating an answer..."):
        answer, retrieved_docs = answer_query(query, vectorstore, llm)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Supporting Policy Sources")

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get(
            "source_file",
            Path(doc.metadata.get("source", "Unknown source")).name,
        )
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")

        with st.expander(f"Source {i}: {source} | {chunk_id}"):
            st.write(doc.page_content)
