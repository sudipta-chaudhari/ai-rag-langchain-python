# Retrieval-Augmented Generation System

A comprehensive Python implementation of a **Retrieval-Augmented Generation (RAG)** pipeline that combines document retrieval with Large Language Models to provide accurate, context-aware answers. This project demonstrates how to build an intelligent question-answering system that can reference your own documents.

## Table of Contents

- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage Guide](#usage-guide)
- [Detailed Module Documentation](#detailed-module-documentation)
- [Project Flow](#project-flow)
- [Dependencies](#dependencies)
- [Features](#features)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

RAG (Retrieval-Augmented Generation) is a powerful AI technique that combines two key components:

1. **Retrieval**: Searching a knowledge base (vector database) to find relevant document chunks
2. **Generation**: Using an LLM to generate answers based on retrieved context

### Why RAG?

Traditional language models have limitations:
- **Knowledge cutoff**: Can only know information from training data
- **Hallucination**: May generate plausible-sounding but incorrect information
- **No context**: Can't refer to your specific documents or data

RAG solves these problems by:
- ✅ Grounding answers in actual documents
- ✅ Reducing hallucinations through factual retrieval
- ✅ Enabling Q&A on custom documents without fine-tuning
- ✅ Keeping information current and verifiable

---

## Project Architecture

```
USER QUERY
    ↓
[INPUT PROCESSING]
    ↓
[EMBEDDING GENERATION]
    ↓
[SIMILARITY SEARCH in Vector Store]
    ↓
[RETRIEVE TOP K DOCUMENTS]
    ↓
[CONTEXT + QUERY to LLM]
    ↓
[LLM GENERATES ANSWER]
    ↓
RESPONSE TO USER
```

### Two Main Phases

#### Phase 1: Ingestion (Offline)
- Load PDF documents from disk
- Split documents into manageable chunks
- Generate vector embeddings for each chunk
- Store embeddings in FAISS vector database

#### Phase 2: Retrieval & Generation (Online)
- Receive user query
- Generate embedding for query
- Search vector store for similar documents
- Combine retrieved context with query
- Generate answer using LLM

---

## Project Structure

```
RAG_Demo_Python/
├── README.md                          # This file - comprehensive documentation
├── pyproject.toml                     # Python project metadata and dependencies
├── data/                              # Folder for input PDF documents
│   └── [Place your PDF files here]    # PDFs to be ingested
├── vector_store/                      # FAISS vector database (auto-generated)
│   ├── index.faiss                    # FAISS index with embeddings
│   └── docstore.pkl                   # Document metadata storage
├── src/                               # Main source code
│   ├── __init__.py                    # Package initialization
│   ├── config.py                      # Configuration parameters (LLM, data paths, etc.)
│   ├── main.py                        # Main entry point with CLI interface
│   ├── ingestion.py                   # Data ingestion pipeline (load PDFs → create embeddings)
│   ├── retrieval.py                   # Query pipeline (search → generation)
└── venv/                              # Python virtual environment (created during setup)
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Where you place PDF files to be ingested into the RAG system |
| `vector_store/` | Persisted FAISS database with embeddings (created after ingestion) |
| `src/` | Core source code for the RAG pipeline |

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- LM-Studio (for local model) OR OpenAI keys (for cloud models)

### Step 1: Clone/Setup Project

```bash
cd RAG_Demo_Python
```

### Step 2: Create Virtual Environment

```bash
# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt

# Or using the pyproject.toml
pip install -e .
```

### Step 4: Configure LLM

#### Option A: Local LLM (Recommended for Privacy & Cost)

1. **Download LM-Studio**: https://lmstudio.ai/
2. **Download a model** (e.g., Nomic Embed Text for embeddings)
3. **Start LM-Studio server**: Click "Start Server" (default: http://127.0.0.1:1234)
4. **Verify connection**: Check that `config.py` has correct `LLM_BASE_URL`

#### Option B: OpenAI API (Cloud)

1. Get API key from https://platform.openai.com/api-keys
2. Create `.env` file in project root:
   ```
   OPENAI_API_KEY=sk-...
   OPENAI_API_URL=https://api.openai.com/v1
   ```

### Step 5: Prepare Data

1. Place PDF files in the `data/` folder
2. Supported formats: PDF (PDFs with text - not scanned images)

---

## Configuration

All configuration is managed in `src/config.py`:

```python
# LLM Configuration
LLM_BASE_URL = "http://127.0.0.1:1234/v1"  # Local LM-Studio endpoint
LLM_MODEL = "text-embedding-nomic-embed-text-v1.5"  # Embedding model
LLM_TEMPERATURE = 0.7  # Randomness (0.0-1.0): lower = focused, higher = creative

# Data Configuration
CHUNK_SIZE = 1000  # Max characters per chunk
CHUNK_OVERLAP = 200  # Overlapping chars between chunks (for context continuity)

# Vector Store Configuration
VECTOR_STORE_PATH = "../vector_store"  # Where to save FAISS database
```

### Parameter Tuning Guide

| Parameter | Impact | Recommended | Notes |
|-----------|--------|-------------|-------|
| `CHUNK_SIZE` | Context per retrieval | 800-1500 | Larger = more context, less precise |
| `CHUNK_OVERLAP` | Context continuity | 100-300 | Higher = better flow, more storage |
| `LLM_TEMPERATURE` | Answer creativity | 0.3-0.7 | Lower for factual, higher for creative |
| Retrieval k | Number of docs | 3-5 | More = broader context, slower |

---

## Usage Guide

### Basic Usage

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the main application
python src/main.py
```

This will:
1. Load and embed all PDFs from `data/` folder
2. Create/update the FAISS vector store
3. Enter interactive query mode

### Interactive Query Mode

```
=== Query Mode ===

Ask a question (or 'exit' to quit): What is machine learning?

Searching...

Answer: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed...

Ask a question (or 'exit' to quit): exit
```

### Programmatic Usage

```python
from src.ingestion import ingest_data
from src.retrieval import query

# Load and embed documents
ingest_data()

# Query the system
answer = query("What is cloud computing?")
print(answer)
```

---

## Detailed Module Documentation

### 1. `src/config.py` - Configuration Module

**Purpose**: Centralizes all configuration parameters

**Key Components**:
- LLM settings (model, API endpoint, temperature)
- Data processing parameters (chunk size, overlap)
- File paths (data folder, vector store location)

**Usage**:
```python
from config import LLM_BASE_URL, CHUNK_SIZE, DATA_FOLDER
```

### 2. `src/ingestion.py` - Data Ingestion Pipeline

**Purpose**: Loads documents and creates embeddings

**Key Functions**:

#### `load_pdfs(data_folder: str) -> list`
- Scans data folder for PDF files
- Uses `PyPDFLoader` to extract text
- Returns list of Document objects
- **Output**: Raw documents with metadata

#### `chunk_documents(documents: list) -> list`
- Splits documents into smaller chunks
- Uses `RecursiveCharacterTextSplitter` for semantic coherence
- Respects paragraph, sentence, and word boundaries
- Adds overlap between chunks for context continuity
- **Output**: Chunked documents ready for embedding

#### `create_vector_store(chunked_docs: list) -> FAISS`
- Generates embeddings using OpenAI-compatible API
- Creates FAISS (Facebook AI Similarity Search) index
- Persists index to disk for later retrieval
- **Output**: Saved FAISS database

#### `ingest_data()`
- Main orchestration function
- Calls load → chunk → embed in sequence
- **Usage**: `ingest_data()`

**Example**:
```python
from src.ingestion import ingest_data

# Run full ingestion pipeline
ingest_data()
# Output: Vector store saved to vector_store/
```

### 3. `src/retrieval.py` - Query & Retrieval Pipeline

**Purpose**: Retrieves relevant documents and generates answers

**Key Functions**:

#### `load_vector_store() -> FAISS`
- Loads persisted FAISS database from disk
- Initializes embeddings for query encoding
- Must use same embedding model as ingestion
- **Output**: FAISS vector store ready for search

#### `create_qa_chain() -> RetrievalQA`
- Creates LangChain RetrievalQA chain
- Configures retriever (k=3 documents)
- Initializes LLM for answer generation
- **Chain type**: "stuff" (concatenates all retrieved docs)
- **Output**: Configured QA chain

#### `query(question: str) -> str`
- Main query function
- Embeds question and searches vector store
- Retrieves top-k similar document chunks
- Generates answer using LLM with context
- **Input**: User question
- **Output**: Generated answer

**Flow**:
```
User Question
    ↓
embed(question)
    ↓
FAISS.similarity_search(query_embedding, k=3)
    ↓
retrieved_documents
    ↓
prompt = question + retrieved_documents
    ↓
LLM.generate(prompt)
    ↓
answer
```

**Example**:
```python
from src.retrieval import query

answer = query("What is the main topic?")
print(answer)
```

### 4. `src/main.py` - Main Entry Point

**Purpose**: Provides CLI interface for the RAG system

**Flow**:
1. Display welcome message
2. Call `ingest_data()` to prepare knowledge base
3. Enter interactive loop
4. Process user queries until 'exit'

**Usage**:
```bash
python src/main.py
```

---

## Project Flow

### Complete RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: DATA INGESTION (Offline)               │
└─────────────────────────────────────────────────────────────┘

1. User places PDF files in data/ folder
         ↓
2. ingest_data() is called
         ↓
3. load_pdfs()
   - Scan data/ for PDF files
   - Use PyPDFLoader to extract text
   - Return: Raw documents with metadata
         ↓
4. chunk_documents()
   - Split documents into chunks (1000 chars max)
   - Add 200 char overlap for context continuity
   - Use recursive splitting at paragraph/sentence/word boundaries
   - Return: Chunked documents
         ↓
5. create_vector_store()
   - Initialize embeddings using LLM
   - Generate vector embedding for each chunk
   - Create FAISS index
   - Save to vector_store/ directory
         ↓
6. Vector store ready for queries
   Status: Knowledge base prepared ✓

┌─────────────────────────────────────────────────────────────┐
│            PHASE 2: QUERY & RETRIEVAL (Online)              │
└─────────────────────────────────────────────────────────────┘

1. User enters question in interactive mode
         ↓
2. query(question) is called
         ↓
3. load_vector_store()
   - Load FAISS from disk
   - Initialize embeddings
   - Return: Loaded vector store
         ↓
4. create_qa_chain()
   - Initialize LLM
   - Configure retriever (k=3)
   - Create RetrievalQA chain
   - Return: Configured chain
         ↓
5. Embedding & Retrieval
   - Embed the user question
   - Search vector store for similar chunks
   - Retrieve top 3 most relevant documents
   - Return: [doc1, doc2, doc3]
         ↓
6. Context Preparation
   - Combine question + retrieved documents
   - Create prompt for LLM
         ↓
7. LLM Generation
   - Send prompt to LLM
   - LLM generates answer based on context
   - Return: Generated answer
         ↓
8. Display answer to user
         ↓
9. Loop back to step 1 for next question
```

### Data Flow Diagram

```
                    ┌─────────────┐
                    │ PDF Files   │
                    │ in data/    │
                    └──────┬──────┘
                           │
                           ↓
                    ┌─────────────────────┐
                    │   PyPDFLoader       │
                    │  (Extract text)     │
                    └────────┬────────────┘
                             │
                             ↓ Raw text chunks
                    ┌─────────────────────────┐
                    │ RecursiveTextSplitter   │
                    │ (1000 chars + 200 over) │
                    └────────┬────────────────┘
                             │
                             ↓ Text chunks
                    ┌─────────────────────────┐
                    │  OpenAI Embeddings      │
                    │ (Via LM-Studio)         │
                    └────────┬────────────────┘
                             │
                             ↓ Vector embeddings
                    ┌─────────────────────────┐
                    │   FAISS Vector Store    │
                    │   (Similarity search)   │
                    └────────┬────────────────┘
                             │
                             ↓ Top k similar docs
                    ┌─────────────────────────┐
                    │  LLM (GPT/Claude/etc)   │
                    │  (Generate answer)      │
                    └────────┬────────────────┘
                             │
                             ↓ Generated Answer
                    ┌─────────────────────────┐
                    │   Response to User      │
                    └─────────────────────────┘
```

---

## Dependencies

### Core Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | ≥0.1.0 | LLM orchestration framework |
| `langchain-community` | ≥0.0.10 | Community integrations (FAISS, PDFLoader) |
| `langchain-openai` | ≥0.3.0 | OpenAI integrations (embeddings, LLM) |
| `langchain-text-splitters` | ≥0.3.0 | Document chunking utilities |
| `faiss-cpu` | ≥1.13.2 | Vector similarity search library |
| `pypdf` | ≥3.17.1 | PDF parsing and extraction |
| `openai` | ≥1.3.0 | OpenAI API client |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `numpy` | ≥2.4.3 | Numerical computing (required by FAISS) |

### Installation

```bash
# From requirements.txt
pip install -r requirements.txt

# Or from pyproject.toml
pip install -e .
```

---

## Features

### Current Features
- ✅ PDF document ingestion
- ✅ Semantic text chunking with overlap
- ✅ Vector embedding generation
- ✅ FAISS vector store for fast similarity search
- ✅ LangChain integration for LLM orchestration
- ✅ Support for multiple LLM backends (local, OpenAI)
- ✅ Interactive CLI query interface
- ✅ Context-aware answer generation
- ✅ Configurable chunk size and overlap
- ✅ Temperature control for answer randomness

### Future Enhancement Ideas
- 🔲 Support for multiple document formats (DOCX, TXT, HTML)
- 🔲 Swap FAISS with pgVector is an open-source extension for PostgreSQL that allows you to store, index & search AI-generated embeddings (vectors) directly inside the database.
- 🔲 Web UI (Streamlit/Gradio)
- 🔲 Hybrid search (BM25 + semantic)
- 🔲 Query expansion and refinement
- 🔲 Document metadata filtering
- 🔲 Answer citation tracking
- 🔲 Conversation history and context preservation
- 🔲 Fine-tuning on domain-specific data
- 🔲 Multi-language support
- 🔲 Performance metrics and logging
- 🔲 Caching for repeated queries

---

## Advanced Configuration

### Adjusting Chunk Parameters

**Smaller chunks (500-800 chars)**:
- ✅ More focused retrieval
- ❌ Less context per chunk
- ✅ Faster processing
- Use for: Dense technical documents

**Larger chunks (1200-2000 chars)**:
- ✅ More context retained
- ✅ Fewer chunks to search
- ❌ May retrieve less relevant information
- Use for: Narrative or story documents

### Adjusting Temperature

```python
LLM_TEMPERATURE = 0.1    # Very deterministic (factual Q&A)
LLM_TEMPERATURE = 0.7    # Balanced (default)
LLM_TEMPERATURE = 1.0    # Very creative (brainstorming)
```

### Adjusting Retrieval Count

In `src/retrieval.py`, change `search_kwargs={"k": 3}`:
```python
retriever=vector_store.as_retriever(search_kwargs={"k": 5})  # Get top 5 docs
```

---

## Troubleshooting

### Issue: "No PDF files found in data folder"

**Solution**:
1. Ensure PDF files are in `data/` folder at project root
2. Check file extensions are `.pdf` (lowercase)
3. Verify PDFs contain extractable text (not scanned images)

### Issue: "Connection refused to LM-Studio"

**Solution**:
1. Start LM-Studio application
2. Click "Start Server" button
3. Verify it shows "http://127.0.0.1:1234/v1"
4. Check `config.py` has correct URL

### Issue: "OPENAI_API_KEY not found"

**Solution**:
1. Create `.env` file in project root
2. Add: `OPENAI_API_KEY=sk-your-key-here`
3. Or set environment variable directly:
   ```bash
   $env:OPENAI_API_KEY = "sk-..."
   ```

### Issue: "Module not found" errors

**Solution**:
```bash
# Verify virtual environment is activated
.\venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -e .
```

### Issue: Slow ingestion with large PDFs

**Solution**:
1. Check if LM-Studio is loaded with model
2. Reduce CHUNK_SIZE in config.py
3. Process PDFs in batches

### Issue: Poor quality answers

**Optimization tips**:
1. Increase `k` (number of retrieved documents) from 3 to 5
2. Reduce CHUNK_SIZE for more targeted retrieval
3. Lower LLM_TEMPERATURE for factual answers (0.3-0.5)
4. Ensure PDFs have good quality text
5. Review retrieved documents with `verbose=True`

---

## Example Queries

### Technical Queries
- "What are the main components?"
- "How does this process work?"
- "What are the key steps?"

### Summarization
- "Summarize the document"
- "What are the key points?"
- "Give me an overview"

### Specific Information
- "What is mentioned about XYZ?"
- "When did this happen?"
- "Who is responsible for this?"

### Comparative
- "What's the difference between X and Y?"
- "Compare the two approaches"
- "Which is better for Z use case?"

---

## Support & Contributions

### Getting Help
1. Check [Troubleshooting](#troubleshooting) section
2. Review inline code comments
3. Check LLM documentation:
   - [LangChain Docs](https://python.langchain.com/)
   - [OpenAI Docs](https://platform.openai.com/docs/)

### Key Resources
- **LM-Studio**: https://lmstudio.ai/
- **LangChain Documentation**: https://python.langchain.com/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Vector Database Guide**: https://www.pinecone.io/learn/vector-database/

---

## License

This project is open-source and available under the MIT License. See LICENSE file for details.

---

### Related Technologies
- Vector databases: FAISS, Pinecone, Weaviate, Milvus
- Text embeddings: Sentence Transformers, OpenAI Embeddings, Cohere
- LLMs: GPT-3/4, Claude, LLaMA, Mistral

---

## Quick Start Summary

```bash
# 1. Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -e .

# 2. Prepare data
# - Place PDFs in data/ folder
# - Start LM-Studio and load a model

# 3. Run
python src/main.py

# 4. Query
Ask a question (or 'exit' to quit): What is the main topic?
```

---

**Last Updated**: March 2026

For the most current information, check inline code comments in each module.
