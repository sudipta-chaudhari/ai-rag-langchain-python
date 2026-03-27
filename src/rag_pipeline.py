"""
RAG Pipeline Class - Object-oriented implementation of the RAG system.

This class encapsulates all functionality of the RAG pipeline using
separate class-based components for configuration, ingestion, and retrieval.
"""

from config import Config
from ingestion import Ingestion
from retrieval import Retrieval


class RAGPipeline:
    """
    A class-based RAG (Retrieval-Augmented Generation) pipeline.

    This class manages the entire RAG workflow by delegating to specialized
    component classes: Config, Ingestion, and Retrieval.
    """

    def __init__(self):
        """Initialize the RAG pipeline with private properties."""
        self._config = Config()
        self._ingestion = Ingestion(self._config)
        self._retrieval = Retrieval(self._config)

    def ingest(self):
        """Run the data ingestion pipeline."""
        self._ingestion.run()

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question (str): The user's question to answer.

        Returns:
            str: The generated answer based on retrieved context.
        """
        return self._retrieval.query(question)
