
# MeridianHealth: Context-Aware RAG Assistant for Insurance Policy Queries

## Project Overview

MeridianHealth is a case-study project that builds a context-aware **Retrieval-Augmented Generation (RAG)** assistant to help health-insurance representatives find and interpret information across policy documents.

Insurance policy information is distributed across benefit guides, exclusions, reimbursement rules, and terms and conditions. A static FAQ can answer known, repetitive questions, but it is less effective when a representative asks a question in different words, combines multiple issues, or needs information that is not covered by a predefined FAQ entry.

This project moves beyond FAQ lookup by retrieving relevant passages from the underlying policy documents and using those passages as evidence for a language-model response. The assistant is designed to provide an answer alongside supporting source information, rather than asking users to rely on an unsupported generated response.

> **Scope note:** MeridianHealth is the scenario used in this project. This README describes the implementation documented in the notebook; it does not claim that the system was deployed in a live insurance business or that business impact was measured.

## Project Aim

Build an end-to-end RAG workflow that can:

- Ingest and structure health-insurance policy PDFs.
- Split policy content into retrieval-friendly chunks while retaining useful document structure.
- Create semantic embeddings and a searchable vector index.
- Retrieve policy passages relevant to a representative's question.
- Generate an answer grounded in retrieved policy context, with instructions not to invent missing policy details.
- Expose the workflow through a reusable local Streamlit inference application.
- Show source filenames and chunk IDs so users can inspect the evidence behind an answer.

## Problem Statement

The original FAQ-based approach is useful for direct questions with known answers, but it has limitations when queries are:

- **Rephrased:** The representative's wording differs from the FAQ wording.
- **Multi-part:** The question asks about several conditions or benefits.
- **Cross-document:** The answer depends on information spread across policy sections or documents.
- **Not covered by the FAQ:** The needed information must be found in the source policies.

The RAG approach searches the underlying policy knowledge base semantically instead of relying only on a predefined list of questions and answers.

## Solution Architecture

```text
Policy PDFs
    ↓
Document parsing with Docling
    ↓
Markdown / structured text extraction
    ↓
Token-aware recursive chunking
    ↓
Domain-specific embeddings
    ↓
FAISS vector index
    ↓
Representative's question
    ↓
Top-k semantic retrieval
    ↓
Retrieved passages + source metadata
    ↓
Prompt augmentation
    ↓
LLM-generated answer grounded in retrieved context
    ↓
Answer + supporting source passages in Streamlit
```

The notebook separates **index preparation** from **query-time inference**. Policy documents are processed and indexed ahead of time. The local Streamlit app loads the saved index rather than rebuilding the knowledge base whenever it starts.

## Technical Implementation

### 1. Document Ingestion and Preparation

- Used **Docling** to convert policy PDFs into a structured document representation and export Markdown.
- Inspected extracted text and tables, recognizing that headings, clauses, and benefit tables carry important meaning in insurance documents.
- Worked with four policy-document categories described in the notebook:
  - `MH_Plan_Benefits_Coverage_Guide.pdf`
  - `MH_Claim_Exclusion_Reference.pdf`
  - `MH_Reimbursement_and_Claims_Procedure.pdf`
  - `MH_Terms_and_Conditions.pdf`
- Kept source metadata with the chunks to support traceability.

### 2. Token-Aware Chunking

Used LangChain's `RecursiveCharacterTextSplitter` with the embedding model's tokenizer and the following configuration:

| Setting | Configuration |
|---|---|
| Chunk-size target | 512 tokens |
| Chunk overlap | 50 tokens |
| Splitting strategy | Recursive, prioritizing paragraph and line/section boundaries |
| Metadata | Source file and chunk ID |

The chunking strategy aims to retain meaningful policy passages while keeping each chunk within the embedding model's stated input limit. The 50-token overlap helps preserve context when a useful statement crosses a chunk boundary.

### 3. Embeddings and Vector Search

- Used the domain-oriented embedding model `surajvbangera/mediclaim_embedding`.
- Normalized embeddings and created a local **FAISS** vector index.
- Implemented semantic similarity retrieval using top-k results, with `k = 3` in the example workflow.
- Persisted the vector index to disk so it can be reloaded for inference.

### 4. Context-Grounded Generation

- Configured `gpt-4o-mini` as the generation model through a LangChain chat-model interface.
- Built a reusable function to format retrieved policy chunks with their source filename and chunk ID.
- Implemented a generation prompt instructing the model to:
  - Answer using only the retrieved policy context.
  - Avoid outside knowledge, assumptions, and invented policy details.
  - Include relevant conditions, exclusions, limits, waiting periods, and exceptions.
  - State when the available policy context is insufficient to answer.
- Combined retrieval and generation into a reusable RAG function that returns both the answer and retrieved documents.

### 5. Streamlit Inference Application

- Created an `app.py` Streamlit application to accept policy questions, load the persisted FAISS index, retrieve relevant passages, generate a grounded response, and display supporting source passages.
- Prepared an application bundle containing `app.py` and the saved FAISS index for local use in VS Code.
- Documented local setup with a Python virtual environment and the required packages.
- Kept API credentials in a separate local `config.json` rather than including secrets in the application bundle.

## What I Achieved

Based on the notebook implementation, the project demonstrates the following technical achievements:

1. **Built a document-to-vector-search pipeline:** Policy PDFs are converted into structured text, chunked with a tokenizer-aware strategy, embedded, and stored in a FAISS index.
2. **Implemented a reusable RAG workflow:** A question retrieves the top three semantically similar policy chunks, adds those passages to the generation prompt, and returns the answer together with the retrieved documents.
3. **Added source traceability:** Retrieved chunks retain source filenames and chunk IDs, and the generation prompt is designed to avoid unsupported answers when the retrieved context is insufficient.
4. **Persisted and reloaded the knowledge base:** The notebook confirms that a saved FAISS index was reloaded with **9 vectors** in the demonstrated run.
5. **Packaged an inference interface:** Created a Streamlit `app.py` and a ZIP bundle containing the application and FAISS index, with documented steps for running it locally.

### Validation and Limitations

The notebook shows that the end-to-end components were assembled and that the saved index could be reloaded. It does not report a formal retrieval benchmark, answer-accuracy score, hallucination rate, latency benchmark, user study, or measured business improvement.

One sample query asking for copay amounts returned that the available policy documents did not contain enough information. This illustrates the configured insufficient-context response, but it is not by itself proof of overall answer quality.

Further evaluation should test retrieval relevance and answer correctness across a labeled set of policy questions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Document processing | Docling |
| LLM orchestration | LangChain, `langchain-openai` |
| Embeddings | Hugging Face, `surajvbangera/mediclaim_embedding` |
| Vector database / index | FAISS |
| Application interface | Streamlit |
| Development environment | Google Colab, VS Code |
| Environment management | Python virtual environment |

## How to Run the Local Application

The notebook documents the following local setup. The required `app.py` and `faiss_index` artifacts must first be generated and extracted from the notebook bundle.

### 1. Set Up the Project Directory

Place `app.py` and the `faiss_index/` directory in the same project folder.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS/Linux:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install streamlit langchain-huggingface langchain-community langchain-openai faiss-cpu transformers sentence-transformers
```

### 4. Configure API Credentials

Create a local `config.json` containing the API key and API base expected by the application.

**Security note:** Never commit API keys, credentials, or other secrets to GitHub. Add `config.json` to `.gitignore` and use environment variables or a secrets manager for a production application.

### 5. Run the Application

```bash
streamlit run app.py
```

The app is documented as a **local inference application**, not a production deployment. Keep `index.faiss` and `index.pkl` together and preserve the directory structure expected by `app.py`.

## Future Improvements

Potential next steps include:

- Build a labeled test set covering direct, rephrased, multi-part, and cross-document questions.
- Evaluate retrieval quality, including whether the correct policy passages appear in the top-k results.
- Evaluate answer correctness against policy-grounded reference answers.
- Test questions involving plan tier, state, policy version, effective date, and conflicting source content.
- Add richer metadata filters and a clear escalation path when evidence is incomplete or conflicting.
- Measure latency and resource use, and validate the app's behavior before considering production use.

## Resume Highlights

- Built an end-to-end **Retrieval-Augmented Generation (RAG) pipeline** in Python to query health-insurance policy PDFs, using Docling for document parsing, tokenizer-aware chunking, Hugging Face embeddings, and FAISS semantic search.
- Integrated **LangChain and GPT-4o-mini** to generate policy-grounded responses from the top three retrieved passages, with prompt instructions to avoid unsupported claims and disclose when context is insufficient.
- Developed and packaged a **Streamlit inference application** with a persistent FAISS knowledge base and source-level traceability, enabling users to inspect the policy passages supporting generated answers.
