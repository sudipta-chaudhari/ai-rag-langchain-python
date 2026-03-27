"""
Data Ingestion Class for the RAG Pipeline.

This class handles the complete data ingestion process:
1. Load PDF documents from the data folder
2. Split documents into manageable chunks
3. Generate embeddings for each chunk
4. Store embeddings in a FAISS vector database
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings_utils import initialize_embeddings
from langchain_community.vectorstores import FAISS


class Ingestion:
    """
    A class-based data ingestion pipeline for RAG.

    This class manages document loading, chunking, and vector store creation.
    """

    def __init__(self, config):
        """
        Initialize the ingestion pipeline.

        Args:
            config: Configuration object with settings for data folder, chunk size,
                   chunk overlap, vector store path, and LLM settings.
        """
        self._config = config
        self._embeddings = None

    def load_pdfs(self) -> list:
        """Load all PDF files from the data folder."""
        documents = []
        pdf_files = list(Path(self._config.data_folder).glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self._config.data_folder}")
            return documents

        for pdf_file in pdf_files:
            print(f"Loading {pdf_file.name}...")
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())

        return documents

    def chunk_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunked_docs = text_splitter.split_documents(documents)
        print(f"Created {len(chunked_docs)} chunks")
        return chunked_docs

    def run(self):
        """
        Execute the complete ingestion pipeline.

        This method:
        1. Loads all PDFs from the data folder
        2. Splits documents into chunks
        3. Creates embeddings and stores them in a FAISS vector database
        4. Saves the vector store to disk
        """
        print("Starting data ingestion...")

        # Load PDFs
        documents = self.load_pdfs()
        if not documents:
            print("No documents to process. Ingestion complete.")
            return

        # Chunk documents
        chunked_docs = self.chunk_documents(documents)

        # Initialize embeddings
        self._embeddings = initialize_embeddings(self._config)

        # Create and save vector store
        print("Creating vector store...")
        vector_store = FAISS.from_documents(chunked_docs, self._embeddings)

        # Ensure directory exists
        os.makedirs(self._config.vector_store_path, exist_ok=True)
        vector_store.save_local(self._config.vector_store_path)
        print(f"Vector store saved to {self._config.vector_store_path}")
        print("Ingestion complete!")
