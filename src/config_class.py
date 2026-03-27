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

    All configuration values are stored as instance attributes.
    """

    def __init__(self):
        """Initialize configuration with default values."""
        # Load environment variables from .env file
        load_dotenv()

        # ==================== LLM Configuration ====================
        self.llm_base_url = "http://127.0.0.1:1234/v1"
        self.llm_api_key = "not needed"
        self.llm_model = "text-embedding-nomic-embed-text-v1.5"
        self.llm_temperature = 0.7

        # ==================== Data Configuration ====================
        self.data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
        self.chunk_size = 1000
        self.chunk_overlap = 200

        # ==================== Vector Store Configuration ====================
        self.vector_store_path = os.path.join(os.path.dirname(__file__), "..", "vector_store")
