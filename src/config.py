"""
Configuration module for the RAG (Retrieval-Augmented Generation) Pipeline.

This module centralizes all configuration parameters including LLM settings,
data processing parameters, and vector store paths.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== LLM (Large Language Model) Configuration ====================
# These settings configure how the LLM is accessed and how it behaves

# BASE URL for the LLM API endpoint (running on local machine)
LLM_BASE_URL = "http://127.0.0.1:1234/v1"

# The specific model to use for embeddings
# This model converts text into vector representations for semantic search
LLM_MODEL = "text-embedding-nomic-embed-text-v1.5"

# Temperature controls randomness in LLM responses (0.0-1.0)
# Lower values (closer to 0) produce more deterministic/focused responses
# Higher values (closer to 1) produce more creative/varied responses
LLM_TEMPERATURE = 0

# ==================== Data Configuration ====================
# These settings control how documents are processed and stored

# Path to the data folder containing PDF files to be ingested
# Uses relative path to locate the 'data' folder at the root level
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "data")

# CHUNK_SIZE: Maximum number of characters in each text chunk
# Larger chunks retain more context but may be less focused
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: Number of overlapping characters between consecutive chunks
# Overlap ensures context continuity between chunks for better retrieval
CHUNK_OVERLAP = 200

# ==================== Vector Store Configuration ====================
# These settings configure the storage and location of vector embeddings

# Path to store the FAISS (Facebook AI Similarity Search) vector database
# FAISS is an efficient library for similarity search and clustering of dense vectors
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vector_store")
