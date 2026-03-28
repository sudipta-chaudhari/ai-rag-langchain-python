"""
Data Ingestion Class for the RAG Pipeline.

This class handles the complete data ingestion process:
1. Load PDF documents from the data folder
2. Split documents into manageable chunks
3. Generate embeddings for each chunk
4. Store embeddings in a FAISS vector database
"""

import os
import logging
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings_utils import initialize_embeddings
from langchain_community.vectorstores import FAISS
from config import Config
from logging_config import setup_logging

# Initialize logger
logger = setup_logging(__name__)


class Ingestion:
    """
    A class-based data ingestion pipeline for RAG.

    This class manages document loading, chunking, and vector store creation.
    """

    def __init__(self, config) -> None:
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
        
        try:
            pdf_files = list(Path(self._config.data_folder).glob("*.pdf"))
            logger.debug(f"Searching for PDF files in: {self._config.data_folder}")

            if not pdf_files:
                logger.warning(f"No PDF files found in {self._config.data_folder}")
                return documents

            logger.info(f"Found {len(pdf_files)} PDF file(s) to process")

            for pdf_file in pdf_files:
                try:
                    logger.info(f"Loading PDF: {pdf_file.name}")
                    loader = PyPDFLoader(str(pdf_file))
                    pdf_documents = loader.load()
                    logger.debug(f"Loaded {len(pdf_documents)} pages from {pdf_file.name}")
                    documents.extend(pdf_documents)
                except FileNotFoundError as e:
                    logger.error(f"PDF file not found: {pdf_file} - {str(e)}")
                except Exception as e:
                    logger.error(f"Error loading PDF {pdf_file.name}: {type(e).__name__}: {str(e)}", exc_info=True)
                    # Continue with next file instead of failing
                    continue

            logger.info(f"Successfully loaded {len(documents)} total pages from all PDFs")
            return documents

        except Exception as e:
            logger.error(f"Critical error in load_pdfs: {type(e).__name__}: {str(e)}", exc_info=True)
            return documents

    def chunk_documents(self, documents: list) -> list:
        """Split documents into chunks."""
        try:
            if not documents:
                logger.warning("No documents provided for chunking")
                return []

            logger.debug(f"Starting chunking process with {len(documents)} documents")
            logger.debug(f"Chunk size: {self._config.chunk_size}, Overlap: {self._config.chunk_overlap}")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunked_docs = text_splitter.split_documents(documents)
            logger.info(f"Successfully created {len(chunked_docs)} chunks from documents")
            logger.debug(f"Chunking complete. Average chunk size: {sum(len(doc.page_content) for doc in chunked_docs) // len(chunked_docs) if chunked_docs else 0} characters")
            
            return chunked_docs

        except ValueError as e:
            logger.error(f"Invalid chunking parameters: {str(e)}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error during document chunking: {type(e).__name__}: {str(e)}", exc_info=True)
            return []

    def run(self) -> None:
        """
        Execute the complete ingestion pipeline.

        This method:
        1. Loads all PDFs from the data folder
        2. Splits documents into chunks
        3. Creates embeddings and stores them in a FAISS vector database
        4. Saves the vector store to disk
        
        Raises:
            Exception: Logs critical errors but does not raise to allow partial recovery.
        """
        try:
            logger.info("=" * 50)
            logger.info("Starting data ingestion pipeline")
            logger.info("=" * 50)

            # Load PDFs
            logger.debug("Step 1: Loading PDF documents")
            documents = self.load_pdfs()
            
            if not documents:
                logger.warning("No documents loaded. Ingestion terminated.")
                return

            # Chunk documents
            logger.debug("Step 2: Chunking documents")
            chunked_docs = self.chunk_documents(documents)
            
            if not chunked_docs:
                logger.warning("No chunks created. Ingestion terminated.")
                return

            # Initialize embeddings
            try:
                logger.debug("Step 3: Initializing embeddings model")
                self._embeddings = initialize_embeddings(self._config)
                logger.info("Embeddings model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embeddings: {type(e).__name__}: {str(e)}", exc_info=True)
                raise

            # Create and save vector store
            try:
                logger.debug("Step 4: Creating FAISS vector store from chunks")
                vector_store = FAISS.from_documents(chunked_docs, self._embeddings)
                logger.info(f"FAISS vector store created with {len(chunked_docs)} chunks")
            except Exception as e:
                logger.error(f"Failed to create vector store: {type(e).__name__}: {str(e)}", exc_info=True)
                raise

            try:
                logger.debug(f"Step 5: Saving vector store to {self._config.vector_store_path}")
                os.makedirs(self._config.vector_store_path, exist_ok=True)
                vector_store.save_local(self._config.vector_store_path)
                logger.info(f"Vector store successfully saved to {self._config.vector_store_path}")
            except IOError as e:
                logger.error(f"Failed to save vector store: {type(e).__name__}: {str(e)}", exc_info=True)
                raise
            except Exception as e:
                logger.error(f"Unexpected error saving vector store: {type(e).__name__}: {str(e)}", exc_info=True)
                raise

            logger.info("=" * 50)
            logger.info("Data ingestion pipeline completed successfully!")
            logger.info("=" * 50)

        except Exception as e:
            logger.error("=" * 50)
            logger.error(f"CRITICAL ERROR: Ingestion pipeline failed: {type(e).__name__}: {str(e)}", exc_info=True)
            logger.error("=" * 50)
            raise
