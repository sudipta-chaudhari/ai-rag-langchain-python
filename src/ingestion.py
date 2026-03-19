"""
Data Ingestion Module for the RAG Pipeline.

This module handles the complete data ingestion process:
1. Load PDF documents from the data folder
2. Split documents into manageable chunks
3. Generate embeddings for each chunk
4. Store embeddings in a FAISS vector database

The ingestion pipeline creates the knowledge base that will be used
for retrieval-augmented generation during query time.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, VECTOR_STORE_PATH, LLM_BASE_URL

def load_pdfs(data_folder: str) -> list:
    """
    Load all PDF files from the specified data folder.
    
    This function scans the data folder for PDF files and loads them using PyPDFLoader.
    Each PDF is parsed into a list of documents with metadata (filename, page number, etc.).
    
    Args:
        data_folder (str): Path to the folder containing PDF files to load.
    
    Returns:
        list: List of Document objects containing text and metadata from all PDFs.
              Returns empty list if no PDFs are found.
    
    Prints:
        - Status message for each PDF being loaded
        - Warning message if no PDFs are found in the folder
    """
    documents = []
    # Find all PDF files in the data folder
    pdf_files = list(Path(data_folder).glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_folder}")
        return documents
    
    # Load each PDF file
    for pdf_file in pdf_files:
        print(f"Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        # extend() adds all documents from this PDF to the main list
        documents.extend(loader.load())
    
    return documents

def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks with overlapping context.
    
    This function uses RecursiveCharacterTextSplitter to intelligently split documents
    at paragraph, sentence, and word boundaries to maintain semantic coherence.
    Overlapping chunks ensure concepts aren't severed between chunks.
    
    Args:
        documents (list): List of Document objects to be chunked.
    
    Returns:
        list: List of chunked Document objects, each with size limited by CHUNK_SIZE.
    
    Prints:
        - Total count of chunks created
    
    Note:
        The recursive splitting strategy (["\\n\\n", "\\n", " ", ""]) ensures:
        - First tries to split on paragraph boundaries (best for semantics)
        - Falls back to sentence boundaries if needed
        - Then word boundaries if sentences are too long
        - Finally character boundaries as last resort
    """
    # Initialize the text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Separators define the splitting strategy in order of preference
        separators=["\n\n", "\n", " ", ""]
    )
    # Split all documents into chunks
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_docs)} chunks")
    return chunked_docs

def create_vector_store(chunked_docs: list):
    """
    Create embeddings for document chunks and store them in FAISS vector database.
    
    This function:
    1. Initializes OpenAI embeddings
    2. Generates vector embeddings for each document chunk
    3. Stores all embeddings in a FAISS vector database
    4. Saves the vector store to disk for later retrieval
    
    Args:
        chunked_docs (list): List of pre-chunked Document objects ready for embedding.
    
    Returns:
        FAISS: The created FAISS vector store object containing all embeddings.
    
    Prints:
        - Status message indicating embedding and vector store creation
        - Path where the vector store is saved
    
    Notes:
        - Uses OpenAI-compatible API format pointing to local instance
        - check_embedding_ctx_length=False bypasses context length validation
        - Vector store is persisted to disk for reuse without re-embedding
    """
    print("Creating embeddings and vector store...")
    
    # Initialize embeddings using OpenAI-compatible interface
    # Points to local  instance running on localhost:1234
    embeddings = OpenAIEmbeddings(
        model=LLM_MODEL,  # The specific embedding model to use
        base_url=LLM_BASE_URL,  # Local endpoint
        api_key="",  # Dummy key (local model doesn't validate)
        check_embedding_ctx_length=False  # Don't check context length limits
    )
    
    # Create vector store from documents
    # FAISS.from_documents handles the embedding generation and storage
    vector_store = FAISS.from_documents(chunked_docs, embeddings)
    
    # Ensure the vector store directory exists
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    # Save the vector store to disk for persistence
    vector_store.save_local(VECTOR_STORE_PATH)
    
    print(f"Vector store saved to {VECTOR_STORE_PATH}")
    return vector_store


def ingest_data():
    """
    Main ingestion pipeline that orchestrates the complete data ingestion process.
    
    This is the primary function to call for loading and processing documents.
    It coordinates three main steps:
    1. Load all PDF files from the data folder
    2. Split documents into semantic chunks
    3. Create and persist vector embeddings
    
    Prints:
        - Status messages at each stage of the process
        - Completion message or warning if no documents found
    
    Usage:
        >>> ingest_data()
        Starting RAG data ingestion...
        Loading pdf_file.pdf...
        Created 45 chunks
        Creating embeddings and vector store...
        Vector store saved to vector_store/
        Ingestion complete!
    """
    print("Starting RAG data ingestion...")
    # Load all PDFs from data folder
    documents = load_pdfs(DATA_FOLDER)
    
    if documents:
        # Process loaded documents
        chunked_docs = chunk_documents(documents)
        # Create and save vector store
        create_vector_store(chunked_docs)
        print("Ingestion complete!")
    else:
        print("No documents to ingest")
