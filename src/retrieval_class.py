"""
Retrieval Class for the RAG Pipeline.

This class handles the query/retrieval phase of the RAG system:
1. Load the FAISS vector store created during ingestion
2. Initialize the LLM for generating responses
3. Create a RetrievalQA chain that combines retrieval with generation
4. Process user queries and return augmented responses
"""

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate


class Retrieval:
    """
    A class-based retrieval and QA pipeline for RAG.

    This class manages vector store loading, LLM initialization, and query processing.
    """

    def __init__(self, config):
        """
        Initialize the retrieval pipeline.

        Args:
            config: Configuration object with LLM settings and vector store path.
        """
        self._config = config
        self._embeddings = None
        self._vector_store = None
        self._llm = None
        self._qa_chain = None

    def _initialize_embeddings(self):
        """Initialize the embedding model (private method)."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self._config.llm_model,
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
                check_embedding_ctx_length=False
            )

    def load_vector_store(self):
        """
        Load the FAISS vector store from disk.

        Initializes embeddings and loads the persisted vector database.
        """
        self._initialize_embeddings()
        self._vector_store = FAISS.load_local(
            self._config.vector_store_path,
            self._embeddings,
            allow_dangerous_deserialization=True
        )

    def _initialize_llm(self):
        """Initialize the LLM (private method)."""
        if self._llm is None:
            self._llm = ChatOpenAI(
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
                model=self._config.llm_model,
                temperature=self._config.llm_temperature
            )

    def _create_qa_chain(self):
        """Create the RetrievalQA chain (private method)."""
        if self._qa_chain is None:
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use only the provided context to answer the question. "
                "If the answer is not in the context, say exactly: Sorry, I don't know the answer to this question."
                "Do not try to make up an answer."),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])

            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=self._vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": chat_prompt}
            )

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question (str): The user's question to answer.

        Returns:
            str: The generated answer based on retrieved context.
        """
        # Lazy initialization
        if self._vector_store is None:
            self.load_vector_store()
        if self._qa_chain is None:
            self._initialize_llm()
            self._create_qa_chain()

        # Execute query
        result = self._qa_chain.invoke({"query": question})
        return result["result"]
