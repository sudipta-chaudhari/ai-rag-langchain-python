# Retrieval-Augmented Generation (RAG) with LangChain, OpenAI and Python

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
RAG_Pipeline/
├── README.md                          # This file - comprehensive documentation
├── pyproject.toml                     # Python project metadata and dependencies
├── main.py                            # Root entry point with CLI interface
├── data/                              # Folder for input PDF documents
│   └── [Place your PDF files here]    # PDFs to be ingested
├── logs/                              # Application logs (auto-generated)
│   └── rag_pipeline.log              # Rolling log file with DEBUG level detail
├── vector_store/                      # FAISS vector database (auto-generated)
│   ├── index.faiss                    # FAISS index with embeddings
│   ├── docstore.pkl                   # Document metadata storage
│   └── index.pkl                      # Index metadata
├── src/                               # Main source code package
│   ├── __init__.py                    # Package initialization and public API
│   ├── config.py                      # Configuration class with LLM/data settings
│   ├── rag_pipeline.py                # RAGPipeline orchestrator class
│   ├── ingestion.py                   # Ingestion class (load PDFs → create embeddings)
│   ├── retrieval.py                   # Retrieval class (search vectors → generate answers)
│   ├── embeddings_utils.py            # Shared embedding utilities
│   ├── logging_config.py              # Centralized logging setup
│   └── main.py                        # Alternate entry point with logging
└── venv/                              # Python virtual environment (created during setup)
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `data/` | Where you place PDF files to be ingested into the RAG system |
| `logs/` | Application logs with rolling file handler (auto-created) |
| `vector_store/` | Persisted FAISS database with embeddings (created after ingestion) |
| `src/` | Core source code with class-based RAG pipeline implementation |

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
# Using pyproject.toml (recommended)
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

All configuration is managed through the `src/config.py` module using a `Config` class. Configuration values are accessed as read-only properties:

```python
from src.config import Config

config = Config()

# LLM Configuration
print(config.llm_base_url)   # "http://127.0.0.1:1234/v1"
print(config.llm_model)      # "text-embedding-nomic-embed-text-v1.5"
print(config.llm_temperature)     # 0.7

# Data Configuration
print(config.chunk_size)     # 1000
print(config.chunk_overlap)  # 200

# Vector Store Configuration
print(config.vector_store_path)  # Path to vector_store directory

# Logging Configuration
print(config.console_logging_enabled)  # False (enable console output toggle)
```

### Modifying Configuration

To modify configuration values, edit the `src/config.py` file and adjust the `__init__` method:

```python
class Config:
    def __init__(self) -> None:
        # ==================== LLM Configuration ====================
        self._llm_base_url = "http://127.0.0.1:1234/v1"
        self._llm_api_key = "not needed"
        self._llm_model = "text-embedding-nomic-embed-text-v1.5"
        self._llm_temperature = 0.7

        # ==================== Data Configuration ====================
        self._chunk_size = 1000  # Modify here
        self._chunk_overlap = 200  # Or here

        # ==================== Logging Configuration ====================
        self._console_logging_enabled = False  # Set to True for console output
```

### Parameter Tuning Guide

| Property Name | Config Type | Default Value | Recommended Range | Impact | Notes |
|---|---|---|---|---|---|
| `chunk_size` | Data Configuration | 1000 | 500-1500 | Context per retrieval | Smaller = focused (technical docs), Larger = more context (narrative) |
| `chunk_overlap` | Data Configuration | 200 | 100-300 | Context continuity | Higher overlap = better flow but more storage |
| `llm_temperature` | LLM Configuration | 0.7 | 0.1-1.0 | Answer creativity | 0.1 = factual, 0.7 = balanced, 1.0 = creative |
| `k` (retrieval count) | Retrieval Setting | 3 | 3-5 | Number of retrieved docs | More docs = broader context but slower queries |

---

## Usage Guide

### Basic Usage

```bash
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the main application
python main.py
```

This will:
1. Load and embed all PDFs from `data/` folder
2. Create/update the FAISS vector store
3. Enter interactive query mode
4. Log all operations to `logs/rag_pipeline.log`

### Interactive Query Mode

```
=== RAG Pipeline Demo ===

API loaded successfully
Starting data ingestion phase...
Loading PDF files from data folder...
Chunking documents...
Creating vector store...
Data ingestion completed

=== Query Mode ===

Ask a question (or 'exit' to quit): What is machine learning?

Searching...

Answer: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed...

Ask a question (or 'exit' to quit): exit

Goodbye!
```

### Programmatic Usage

```python
from src.rag_pipeline import RAGPipeline

# Create pipeline instance
pipeline = RAGPipeline()

# Run ingestion phase
pipeline.ingest()

# Query the system
answer = pipeline.query("What is cloud computing?")
print(answer)
```

---

## Detailed Module Documentation

### 1. `src/config.py` - Configuration Module

**Purpose**: Centralizes all configuration parameters using a class-based approach

**Key Components**:
- `Config` class: Container for all configuration values with read-only properties
- LLM settings (base URL, API key, model name, temperature)
- Data processing parameters (chunk size, overlap)
- File paths (data folder, vector store location, logs directory)
- Console logging toggle

**Configuration Properties**:
```python
llm_base_url          # Local LM-Studio endpoint (default: http://127.0.0.1:1234/v1)
llm_api_key           # API key for LLM service
llm_model             # Embedding model name
llm_temperature       # Temperature for generation (0.0-1.0)
data_folder           # Path to input PDFs
chunk_size            # Max characters per chunk (default: 1000)
chunk_overlap         # Overlap between chunks (default: 200)
vector_store_path     # Path to FAISS database
console_logging_enabled  # Toggle console logging output
```

**Usage**:
```python
from src.config import Config

config = Config()
print(config.chunk_size)  # 1000
print(config.llm_model)   # text-embedding-nomic-embed-text-v1.5
```

### 2. `src/ingestion.py` - Data Ingestion Pipeline

**Purpose**: Loads documents and creates embeddings through a class-based pipeline

**Ingestion Class Methods**:

#### `__init__(config: Config) -> None`
- Initializes the ingestion pipeline with configuration
- Sets up internal state for embeddings and vector operations

#### `load_pdfs() -> list`
- Scans data folder for PDF files
- Uses `PyPDFLoader` to extract text
- Returns list of Document objects
- **Logging**: Logs each PDF processed and total pages extracted
- **Error Handling**: Continues with next file if one fails

#### `chunk_documents(documents: list) -> list`
- Splits documents into smaller chunks using `RecursiveCharacterTextSplitter`
- Respects semantic boundaries (paragraphs, sentences, words)
- Maintains overlap between chunks for context continuity
- Returns chunked documents ready for embedding
- **Logging**: Logs chunk count and statistics

#### `create_vector_store(chunked_docs: list) -> FAISS`
- Generates embeddings using LLM (via embeddings_utils)
- Creates FAISS index for similarity search
- Persists index to disk
- Returns initialized FAISS vector store
- **Error Handling**: Handles initialization and serialization errors

#### `run() -> None`
- Main orchestration method
- Calls load → chunk → embed in sequence
- Logs all phases and handles errors gracefully

**Example**:
```python
from src.config import Config
from src.ingestion import Ingestion

config = Config()
ingestion = Ingestion(config)
ingestion.run()
# Output: Vector store saved to vector_store/
```

### 3. `src/retrieval.py` - Query & Retrieval Pipeline

**Purpose**: Retrieves relevant documents and generates answers through a class-based pipeline

**Retrieval Class Methods**:

#### `__init__(config: Config) -> None`
- Initializes the retrieval pipeline with configuration
- Sets up internal state for embeddings, LLM, and QA chain

#### `load_vector_store() -> None`
- Loads persisted FAISS database from disk
- Initializes embeddings for query encoding
- Must use same embedding model as ingestion
- **Error Handling**: Raises FileNotFoundError if vector store not found
- **Logging**: Logs each step of loading process

#### `initialize_llm() -> None`
- Initializes ChatOpenAI LLM for answer generation
- Uses configuration for base URL, model, and temperature
- **Error Handling**: Handles connection and configuration errors
- **Caching**: Skips re-initialization if LLM already loaded

#### `create_qa_chain() -> None`
- Creates LangChain RetrievalQA chain
- Configures retriever (k=3 documents by default)
- Sets up prompt template for context-aware generation
- Chain type: "stuff" (concatenates all retrieved docs)
- **Logging**: Logs chain initialization

#### `query(question: str) -> str`
- Main query function
- Embeds question and searches vector store
- Retrieves top-k similar document chunks
- Generates answer using LLM with context
- **Input**: User question
- **Output**: Generated answer
- **Logging**: Logs search results and generation steps

**Flow**:
```
User Question
    ↓
embed(question) using same embeddings as ingestion
    ↓
FAISS.similarity_search(query_embedding, k=3)
    ↓
retrieved_documents (top 3 most similar chunks)
    ↓
prompt = question + context from retrieved_documents
    ↓
LLM.generate(prompt)
    ↓
answer
```

**Example**:
```python
from src.config import Config
from src.retrieval import Retrieval

config = Config()
retrieval = Retrieval(config)
retrieval.load_vector_store()
retrieval.initialize_llm()
retrieval.create_qa_chain()

answer = retrieval.query("What is the main topic?")
print(answer)
```

### 4. `src/rag_pipeline.py` - RAG Pipeline Orchestrator

**Purpose**: Main orchestrator class that coordinates the entire RAG workflow

**RAGPipeline Class**:

#### `__init__() -> None`
- Initializes Config, Ingestion, and Retrieval components
- Sets up the complete pipeline

#### `ingest() -> None`
- Orchestrates the data ingestion phase
- Loads PDFs, chunks them, and creates vector store
- Delegates to Ingestion class

#### `query(question: str) -> str`
- Main entry point for querying the system
- Delegates to Retrieval class
- Returns generated answer

**Usage**:
```python
from src.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()
pipeline.ingest()           # Load and embed documents
answer = pipeline.query("Question here?")
print(answer)
```

### 5. `src/logging_config.py` - Logging Infrastructure

**Purpose**: Centralized logging setup for consistent logging across all modules

**Key Features**:
- Dual handler setup: Console and File
- Console output controlled by `Config.console_logging_enabled`
- File logging always enabled with `RotatingFileHandler`
- Log rotation: 10MB per file with 5 backups
- Logs directory: `logs/rag_pipeline.log`
- Consistent format: `timestamp - logger_name - level - function_name:line_number - message`

**Function**:

#### `setup_logging(logger_name: str = "rag_pipeline", log_level: int = logging.INFO) -> logging.Logger`
- Configures and returns a logger with the specified name
- Creates logs directory if it doesn't exist
- Sets up both console and file handlers
- Handles circular imports by importing Config inside function

**Usage**:
```python
from src.logging_config import setup_logging

logger = setup_logging(__name__)
logger.info("Application started")
logger.debug("Detailed debugging information")
logger.error("An error occurred", exc_info=True)
```

### 6. `src/embeddings_utils.py` - Shared Utilities

**Purpose**: Shared utilities for embedding initialization, used by both ingestion and retrieval

**Key Function**:

#### `initialize_embeddings(config: Config) -> OpenAIEmbeddings`
- Initializes OpenAI-compatible embedding model
- Uses settings from Config class
- Returns initialized OpenAIEmbeddings instance
- Disables embedding context length check for flexibility

**Usage**:
```python
from src.config import Config
from src.embeddings_utils import initialize_embeddings

config = Config()
embeddings = initialize_embeddings(config)
# embeddings ready for use in vector store operations
```

### 7. `main.py` - Main Entry Point

**Purpose**: Provides CLI interface for the RAG system with comprehensive logging

**Flow**:
1. Display welcome message
2. Initialize RAGPipeline
3. Call ingestion pipeline to load and embed documents
4. Enter interactive loop for user queries
5. Handle errors with user-friendly messages
6. Log all activities to file and optionally console

**Usage**:
```bash
python main.py
```

**Features**:
- Logging of all pipeline phases
- Query counter for session statistics
- Proper error handling and reporting
- Graceful exit on Ctrl+C or 'exit' command

---

## Project Flow

### Complete RAG Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: DATA INGESTION (Offline)              │
└─────────────────────────────────────────────────────────────┘

1. User runs: python main.py
   │
   ├─ Setup logging (console + file with rotation)
   │
   └─ Create RAGPipeline() instance
      ├─ Initialize Config (load settings from environment)
      ├─ Initialize Ingestion(config)
      └─ Initialize Retrieval(config)
         │
         └─ Call pipeline.ingest()
            │
            ├─ Ingestion.run()
            │  │
            │  ├─ load_pdfs()
            │  │   - Scan data/ for PDF files
            │  │   - Use PyPDFLoader to extract text
            │  │   - Log each file processed
            │  │   - Return: Raw documents with metadata
            │  │
            │  ├─ chunk_documents()
            │  │   - Split documents into chunks (1000 chars max)
            │  │   - Add 200 char overlap for context continuity
            │  │   - Use recursive splitting (paragraph/sentence/word)
            │  │   - Return: Chunked documents
            │  │
            │  └─ create_vector_store()
            │      - Initialize embeddings using LLM-Studio
            │      - Generate vector embedding for each chunk
            │      - Create FAISS index
            │      - Save to vector_store/ directory
            │      - Log all operations
            │
            └─ Status: Knowledge base prepared ✓
            
            ├─ Logs written to: logs/rag_pipeline.log
            └─ Vector store persisted to: vector_store/

┌─────────────────────────────────────────────────────────────┐
│            PHASE 2: QUERY & RETRIEVAL (Online)              │
└─────────────────────────────────────────────────────────────┘

1. Enter interactive query mode
   │
   └─ Loop: while user != 'exit'
      │
      ├─ Get user query: "What is...?"
      │
      └─ Call pipeline.query(question)
         │
         ├─ Retrieval.load_vector_store()
         │  │ (only on first query)
         │  ├─ Initialize embeddings
         │  └─ Load FAISS from disk
         │
         ├─ Retrieval.initialize_llm()
         │  │ (only on first query)
         │  ├─ Initialize ChatOpenAI
         │  └─ Set temperature and model
         │
         ├─ Retrieval.create_qa_chain()
         │  │ (only on first query)
         │  ├─ Configure retriever (k=3 documents)
         │  └─ Setup context + query template
         │
         ├─ Retrieval.query(question)
         │  │
         │  ├─ Embed the user question
         │  ├─ Search vector store for similar chunks
         │  ├─ Retrieve top 3 most relevant documents
         │  │
         │  ├─ Create context prompt:
         │  │   "Question: {question}\n\nContext: {retrieved_docs}"
         │  │
         │  ├─ Send prompt to LLM
         │  └─ LLM generates answer based on context
         │
         ├─ Return: Generated answer
         │
         ├─ Display answer to user
         │
         ├─ Log query and response
         │
         └─ Loop back for next question
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

| Package | Version | Required | Category | Purpose | Installation |
|---------|---------|----------|----------|---------|--------------|
| `langchain` | ≥0.1.0 | ✅ Yes | Framework | LLM orchestration and chain composition | `pip install langchain` |
| `langchain-community` | ≥0.0.10 | ✅ Yes | Integrations | Community integrations (FAISS, PDFLoader, etc.) | `pip install langchain-community` |
| `langchain-openai` | ≥0.3.0 | ✅ Yes | Integrations | OpenAI embeddings and LLM integration | `pip install langchain-openai` |
| `langchain-text-splitters` | ≥0.3.0 | ✅ Yes | Text Processing | Semantic document chunking utilities | `pip install langchain-text-splitters` |
| `faiss-cpu` | ≥1.13.2 | ✅ Yes | Vector DB | Facebook AI Similarity Search for embeddings | `pip install faiss-cpu` |
| `pypdf` | ≥3.17.1 | ✅ Yes | Text Extraction | PDF text extraction and parsing | `pip install pypdf` |
| `openai` | ≥1.3.0 | ✅ Yes | LLM API | OpenAI API client for models and embeddings | `pip install openai` |
| `python-dotenv` | ≥1.0.0 | ⚠️ Optional | Configuration | Load environment variables from .env files | `pip install python-dotenv` |
| `numpy` | ≥2.4.3 | ✅ Yes | Dependencies | Numerical computing (required by FAISS) | Auto-installed with faiss-cpu |

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
- 🔲 Web UI (REACT / Angular)
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

To modify chunk size and overlap, edit `src/config.py`:

```python
class Config:
    def __init__(self) -> None:
        # Smaller chunks (500-800 chars) for dense technical documents
        self._chunk_size = 500         # More focused retrieval
        self._chunk_overlap = 100       # Less context per chunk
        
        # OR Larger chunks (1200-2000 chars) for narrative documents
        self._chunk_size = 1500        # More context retained
        self._chunk_overlap = 300       # Better continuity
```

**Trade-offs**:
- **Smaller chunks (500-800)**: ✅ More focused retrieval, ✅ Faster, ❌ Less context
- **Larger chunks (1200-2000)**: ✅ More context, ✅ Fewer chunks, ❌ Less precision

### Adjusting Temperature

Modify the `llm_temperature` property in `src/config.py`:

```python
class Config:
    def __init__(self) -> None:
        self._llm_temperature = 0.1    # Very deterministic (factual Q&A)
        # OR
        self._llm_temperature = 0.7    # Balanced (default)
        # OR
        self._llm_temperature = 1.0    # Very creative (brainstorming)
```

### Adjusting Retrieval Count (k)

To retrieve more documents for context, modify `src/retrieval.py` in the `create_qa_chain()` method:

```python
def create_qa_chain(self) -> None:
    # ...
    retriever = self._vector_store.as_retriever(
        search_kwargs={"k": 5}  # Change from 3 to 5 documents
    )
    # ...
```

### Enabling Console Logging

To see logs in the console while running, edit `src/config.py`:

```python
class Config:
    def __init__(self) -> None:
        self._console_logging_enabled = True  # Change from False
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
2. Reduce `chunk_size` in `src/config.py` (e.g., from 1000 to 500)
3. Process PDFs in batches

### Issue: Poor quality answers

**Optimization tips**:
1. Increase `k` (number of retrieved documents) from 3 to 5 in `retrieval.py`
2. Reduce `chunk_size` for more targeted retrieval in `config.py`
3. Lower `llm_temperature` for factual answers (0.3-0.5) in `config.py`
4. Ensure PDFs have good quality text
5. Enable console logging with `config.console_logging_enabled = True` to review details

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
# - Start LM-Studio and load a model (or configure OpenAI API)

# 3. Run
python main.py

# 4. Query
Ask a question (or 'exit' to quit): What is the main topic?
```

---

**Last Updated**: March 2026

For the most current information, check inline code comments in each module.

## Version History

**Current Version**: 0.2.0 (March 2026)
- ✅ Refactored to class-based architecture (Config, Ingestion, Retrieval, RAGPipeline)
- ✅ Added comprehensive logging with file rotation
- ✅ Enhanced error handling across all modules
- ✅ Improved import structure with relative imports
- ✅ Added logging_config.py for centralized logging setup
- ✅ Made configuration values read-only properties

**Previous Version**: 0.1.0
- Basic RAG pipeline with functional approach
- Simple in-memory logging
