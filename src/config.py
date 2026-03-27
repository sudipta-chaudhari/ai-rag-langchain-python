"""
Configuration class for the RAG (Retrieval-Augmented Generation) Pipeline.

This class centralizes all configuration parameters including LLM settings,
data processing parameters, and vector store paths.
"""

import os
from dotenv import load_dotenv


class Config:
    """
    Configuration container for RAG pipeline settings.

    All configuration values are read-only properties.
    """

    def __init__(self) -> None:
        """Initialize configuration with default values."""
        # reads key-value pairs from a .env file and adds them to your system's 
        # environment variables so they can be accessed via os.environ
        load_dotenv()

        # ==================== LLM Configuration ====================
        self._llm_base_url = "http://127.0.0.1:1234/v1"
        self._llm_api_key = "not needed"
        self._llm_model = "text-embedding-nomic-embed-text-v1.5"
        self._llm_temperature = 0.7

        # ==================== Data Configuration ====================
        self._data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
        self._chunk_size = 1000
        self._chunk_overlap = 200

        # ==================== Vector Store Configuration ====================
        self._vector_store_path = os.path.join(os.path.dirname(__file__), "..", "vector_store")

    @property
    def llm_base_url(self) -> str:
        return self._llm_base_url

    @property
    def llm_api_key(self) -> str:
        return self._llm_api_key

    @property
    def llm_model(self) -> str:
        return self._llm_model

    @property
    def llm_temperature(self) -> float:
        return self._llm_temperature

    @property
    def data_folder(self) -> str:
        return self._data_folder

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @property
    def vector_store_path(self) -> str:
        return self._vector_store_path
