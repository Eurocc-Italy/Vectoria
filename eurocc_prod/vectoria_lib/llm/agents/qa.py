#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.llm.helpers import format_docs, get_prompt
from vectoria_lib.db_management.retriever.faiss_retriever import Retriever
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
        retriever = rag_retriever.as_langchain_retriever()
        inference_engine = inference_engine.as_langchain_llm()
        
        chain = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | inference_engine
            | CustomResponseParser()
        )
        self.chain = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=chain)


    def ask(self, question) -> tuple[str,str,str]:
        """
        Ask the QAAgent a question and get an answer based on the retrieved documents 
        and generated response.

        Parameters:
        - question (str): The question to ask the agent.

        Returns:
        - The question, the generated answer and the retrived context.
        """
        output = self.chain.invoke(question)
        return output["question"], output["answer"], output["context"]
