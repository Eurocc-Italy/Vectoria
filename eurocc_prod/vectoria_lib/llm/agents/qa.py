#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import Retriever, FaissRetriever
from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.llm.helpers import format_docs, get_prompt
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class QAAgent:
    """
    A Question Answering (QA) Agent that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The agent can generate answers using a language model and 
    return a response based on the retrieved documents.
    """

    def __init__(
        self,
        rag_retriever: Retriever,
        inference_engine: InferenceEngineBase
    ):
        """
        Initialize the QAAgent object with the provided retriever and inference engine.

        Parameters:
        - rag_retriever (Retriever): A retriever object to fetch relevant documents from a FAISS-based vector store.
        - inference_engine (InferenceEngineBase): An inference engine object to generate answers based on the retrieved documents.
        """

        self.logger = logging.getLogger('llm')
        
        prompt = ChatPromptTemplate.from_template(get_prompt())
        langchain_retriever = rag_retriever.as_langchain_retriever()
        langchain_inference_engine = inference_engine.as_langchain_llm()
        
        self.qa_chain = (
            {"context": langchain_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | langchain_inference_engine
            | CustomResponseParser()
        )


    def ask(self, question) -> str:
        """
        Ask the QAAgent a question and get an answer based on the retrieved documents 
        and generated response.

        Parameters:
        - question (str): The question to ask the agent.

        Returns:
        - str: The generated answer to the question.
        """
        output = self.qa_chain.invoke(question)
        self.logger.info("\n\n=================> Answer:\n%s", output)
        return output
