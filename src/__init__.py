"""
RAG Pipeline Package - Retrieval-Augmented Generation System

This package implements a complete RAG (Retrieval-Augmented Generation) pipeline
that combines document retrieval with Large Language Models for intelligent
question-answering over custom document collections.

Package Modules:
    config: Configuration parameters (LLM settings, data paths, chunk sizes)
    ingestion: Data ingestion pipeline (load PDFs → create embeddings)
    retrieval: Query pipeline (search vectors → generate answers)
    main: CLI interface for the RAG system
    chat: Example implementations using various LLM backends

Usage:
    >>> from ingestion import ingest_data
    >>> from retrieval import query
    >>> ingest_data()
    >>> answer = query("What is the main topic?")

For detailed documentation, see README.md
"""

__version__ = "0.1.0"
__author__ = "RAG Demo Team"
__description__ = "Simple RAG pipeline using LangChain and local LLM"

