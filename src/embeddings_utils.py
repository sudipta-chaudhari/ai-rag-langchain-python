"""Shared embedding initialization utility."""

from langchain_openai.embeddings import OpenAIEmbeddings


def initialize_embeddings(config):
    """Return an OpenAIEmbeddings instance configured from Config object."""
    return OpenAIEmbeddings(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        check_embedding_ctx_length=False
    )
