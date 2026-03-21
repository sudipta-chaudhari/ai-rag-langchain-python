"""
Retrieval Module for the RAG Pipeline.

This module handles the query/retrieval phase of the RAG system:
1. Load the FAISS vector store created during ingestion
2. Initialize the LLM for generating responses
3. Create a RetrievalQA chain that combines retrieval with generation
4. Process user queries and return augmented responses

The retrieval pipeline uses semantic similarity search to find relevant
document chunks, then passes them to the LLM as context for answer generation.
"""

from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from config import VECTOR_STORE_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE

def load_vector_store():
    """
    Load the FAISS vector store that was created during data ingestion.
    
    This function initializes the embedding model and loads the persisted
    FAISS vector database from disk. The vector store contains all the
    embedded document chunks ready for semantic similarity search.
    
    Returns:
        FAISS: The loaded vector store object configured with embeddings.
    
    Notes:
        - Ensures the same embedding model is used as during ingestion
        - allow_dangerous_deserialization=True permits loading external data
    """
    # Initialize embeddings using the same model and settings as ingestion
    # This ensures consistency between embedding and retrieval phases
    embeddings = OpenAIEmbeddings(
        model=LLM_MODEL,  # Must match the model used during ingestion
        base_url=LLM_BASE_URL,  # Local endpoint
        api_key=LLM_API_KEY,  # Dummy key for local model
        check_embedding_ctx_length=False  # Bypass context length validation
    )

    # Load the vector store from disk
    # Uses the same VECTOR_STORE_PATH where it was saved during ingestion
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH, 
        embeddings,
        allow_dangerous_deserialization=True)  # Allow loading external data
    
    return vector_store

def create_qa_chain():
    """
    Create a RetrievalQA chain that combines document retrieval with LLM generation.
    
    This function creates the main RAG chain that:
    1. Takes a user question and converts it to embeddings
    2. Searches the vector store for semantically similar document chunks
    3. Passes the retrieved chunks as context to the LLM
    4. Generates an answer grounded in the retrieved context
    
    Returns:
        RetrievalQA: A fully configured QA chain ready for querying.
    
    Chain Type: "stuff"
        The "stuff" chain type means all retrieved documents are combined
        and passed directly to the LLM. (Alternative: "map_reduce" for larger contexts)
    
    Search Parameters:
        k=3: Retrieve the top 3 most similar document chunks for context.
             This balances relevance with context window limitations.
    """
    # Load the persistent vector store
    vector_store = load_vector_store()
    
    # Initialize the LLM for generating answers
    # Uses OpenAI-compatible API format pointing to local model
    llm = ChatOpenAI(
        base_url=LLM_BASE_URL,  # Local endpoint
        api_key=LLM_API_KEY,  # Dummy key for local model
        model=LLM_MODEL,  # The model to use for generation
        temperature=LLM_TEMPERATURE  # Controls creativity (0.7 = balanced)
    )
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use only the provided context to answer the question. "
        "If the answer is not in the context, say exactly: Sorry, I don't know the answer to this question."
        "Do not try to make up an answer."),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    # Create the RetrievalQA chain with custom prompt
    # This combines the vector store retriever with the LLM for Q&A
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Combine all retrieved docs and send to LLM
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # Top 3 similar chunks
        chain_type_kwargs={"prompt": chat_prompt}  # Use custom prompt template
    )
    
    return qa_chain

def query(question: str) -> str:
    """
    Query the RAG pipeline with a user question.
    
    This is the main function to call when a user submits a question.
    It orchestrates the entire RAG process:
    1. Creates a new QA chain
    2. Embeds the question and searches the vector store
    3. Retrieves relevant document chunks
    4. Generates an answer using the LLM with retrieved context
    
    Args:
        question (str): The user's question to be answered by the RAG system.
    
    Returns:
        str: The generated answer string based on retrieved documents and LLM generation.
    
    Example:
        >>> answer = query("What is machine learning?")
        >>> print(answer)
        "Machine learning is a subset of artificial intelligence..."
    """
    # Create a new QA chain for this query
    qa_chain = create_qa_chain()
    
    # Invoke the chain with the user's question
    # The chain handles embedding, retrieval, and generation internally
    result_dict = qa_chain.invoke({"query": question})
    
    # Extract the generated answer from the result dictionary
    response = result_dict["result"]
    
    return response
