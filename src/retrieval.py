"""
Retrieval Class for the RAG Pipeline.

This class handles the query/retrieval phase of the RAG system:
1. Load the FAISS vector store created during ingestion
2. Initialize the LLM for generating responses
3. Create a RetrievalQA chain that combines retrieval with generation
4. Process user queries and return augmented responses
"""

import logging
from embeddings_utils import initialize_embeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from logging_config import setup_logging

# Initialize logger
logger = setup_logging(__name__)


class Retrieval:
    """
    A class-based retrieval and QA pipeline for RAG.

    This class manages vector store loading, LLM initialization, and query processing.
    """

    def __init__(self, config) -> None:
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

    def load_vector_store(self) -> None:
        """
        Load the FAISS vector store from disk.

        Initializes embeddings and loads the persisted vector database.
        
        Raises:
            FileNotFoundError: If vector store files are not found.
            Exception: If embeddings initialization or vector store loading fails.
        """
        try:
            logger.debug("Loading embeddings model...")
            self._embeddings = initialize_embeddings(self._config)
            logger.info("Embeddings model loaded successfully")
            
            logger.debug(f"Attempting to load FAISS vector store from: {self._config.vector_store_path}")
            self._vector_store = FAISS.load_local(
                self._config.vector_store_path,
                self._embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("FAISS vector store loaded successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Vector store not found at {self._config.vector_store_path}: {str(e)}", exc_info=True)
            raise FileNotFoundError(
                f"Vector store not found. Please run ingestion first. "
                f"Expected location: {self._config.vector_store_path}"
            ) from e
        except ValueError as e:
            logger.error(f"Invalid vector store format or deserialization error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to load vector store: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def initialize_llm(self) -> None:
        """
        Initialize the LLM (Language Model).
        
        Raises:
            ConnectionError: If unable to connect to the LLM service.
            ValueError: If LLM configuration is invalid.
            Exception: If LLM initialization fails.
        """
        try:
            if self._llm is not None:
                logger.debug("LLM already initialized, skipping re-initialization")
                return
            
            logger.debug(f"Initializing LLM with base_url: {self._config.llm_base_url}")
            logger.debug(f"LLM model: {self._config.llm_model}, temperature: {self._config.llm_temperature}")
            
            self._llm = ChatOpenAI(
                base_url=self._config.llm_base_url,
                api_key=self._config.llm_api_key,
                model=self._config.llm_model,
                temperature=self._config.llm_temperature
            )
            logger.info("LLM initialized successfully")
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to LLM service at {self._config.llm_base_url}: {str(e)}", exc_info=True)
            raise
        except ValueError as e:
            logger.error(f"Invalid LLM configuration: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def create_qa_chain(self) -> None:
        """
        Create the RetrievalQA chain for question answering.
        
        Raises:
            TypeError: If LLM or vector store is not properly initialized.
            Exception: If QA chain creation fails.
        """
        try:
            if self._qa_chain is not None:
                logger.debug("QA chain already created, skipping re-creation")
                return
            
            if self._llm is None:
                raise TypeError("LLM must be initialized before creating QA chain")
            if self._vector_store is None:
                raise TypeError("Vector store must be loaded before creating QA chain")
            
            logger.debug("Creating chat prompt template")
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use only the provided context to answer the question. "
                "If the answer is not in the context, say exactly: Sorry, I don't know the answer to this question."
                "Do not try to make up an answer."),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])

            logger.debug("Creating RetrievalQA chain with 'stuff' chain type and k=3 search results")
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=self._llm,
                chain_type="stuff",
                retriever=self._vector_store.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": chat_prompt}
            )
            logger.info("QA chain created successfully")
            
        except TypeError as e:
            logger.error(f"Invalid chain setup: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Failed to create QA chain: {type(e).__name__}: {str(e)}", exc_info=True)
            raise

    def query(self, question: str) -> str:
        """
        Query the RAG pipeline with a question.

        Args:
            question (str): The user's question to answer.

        Returns:
            str: The generated answer based on retrieved context.
            
        Raises:
            ValueError: If question is empty or invalid.
            FileNotFoundError: If vector store is not found.
            Exception: If query execution fails.
        """
        try:
            if not question or not isinstance(question, str):
                logger.warning("Invalid question provided: must be a non-empty string")
                raise ValueError("Question must be a non-empty string")
            
            logger.info(f"Processing query: {question[:100]}{'...' if len(question) > 100 else ''}")

            # Lazy initialization - load vector store if not already loaded
            if self._vector_store is None:
                logger.debug("Vector store not loaded, loading now...")
                self.load_vector_store()
            
            # Initialize LLM if not already done
            if self._llm is None:
                logger.debug("LLM not initialized, initializing now...")
                self.initialize_llm()
            
            # Create QA chain if not already done
            if self._qa_chain is None:
                logger.debug("QA chain not created, creating now...")
                self.create_qa_chain()

            # Execute query
            logger.debug(f"Executing query through QA chain")
            result = self._qa_chain.invoke({"query": question})
            
            if not result or "result" not in result:
                logger.warning("Query returned empty or malformed result")
                return ""
            
            answer = result["result"]
            logger.info(f"Query completed successfully. Answer length: {len(answer)} characters")
            logger.debug(f"Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            return answer

        except ValueError as e:
            logger.error(f"Invalid input: {str(e)}", exc_info=True)
            raise
        except FileNotFoundError as e:
            logger.error(f"Required resource not found: {str(e)}", exc_info=True)
            raise
        except ConnectionError as e:
            logger.error(f"Connection error during query execution: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during query execution: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
