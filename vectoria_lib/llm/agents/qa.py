#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging
from typing_extensions import Annotated, TypedDict
from typing import Sequence

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langsmith import traceable

from vectoria_lib.llm.agents.stateful_workflow import StatefulWorkflow
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.llm.helpers import format_docs
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from vectoria_lib.llm.prompts.prompt_builder import PromptBuilder
class QAAgent:
    """
    A Question Answering (QA) Agent that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The agent can generate answers using a language model and 
    return a response based on the retrieved documents.
    """

    def __init__(
        self,
        rag_retriever: FaissRetriever,
        inference_engine: InferenceEngineBase,
        chat_history: bool,
        system_prompts_lang: str
    ):
        """
        Initialize the QAAgent object with the provided retriever and inference engine.

        Parameters:
        - rag_retriever (Retriever): A retriever object to fetch relevant documents from a FAISS-based vector store.
        - inference_engine (InferenceEngineBase): An inference engine object to generate answers based on the retrieved documents.
        - use_chat_history (bool): A flag to indicate whether to use chat history to contextualize the answers.
        - system_prompts_lang (str): The language of the prompts to load.
        """

        self.logger = logging.getLogger('llm')
        
        retriever = rag_retriever.as_langchain_retriever()
        inference_engine = inference_engine.as_langchain_llm()

        self.use_chat_history = chat_history

        prompt_builder = PromptBuilder(system_prompts_lang)
        
        if self.use_chat_history:

            combine_docs_chain = create_stuff_documents_chain(
                inference_engine,
                prompt_builder.get_qa_prompt_with_history(),
                output_parser=CustomResponseParser()
            )

            history_aware_retriever = create_history_aware_retriever(
                inference_engine,
                retriever,
                prompt_builder.get_contextualize_q_prompt()
            )

            self.rag_chain = create_retrieval_chain(
                history_aware_retriever,
                combine_docs_chain
            )
            self.rag_chain = StatefulWorkflow.to_stateful_workflow(self.rag_chain)

        else:
            combine_docs_chain = create_stuff_documents_chain(
                inference_engine,
                prompt_builder.get_qa_prompt(),
                output_parser=CustomResponseParser()
            )

            self.rag_chain = create_retrieval_chain(
                retriever,
                combine_docs_chain
            )

    # --------------------------------------------------------------------------------
    @traceable
    def ask(self, question: str, session_id: str = None) -> str:
        """
        Ask the QAAgent a input and get an answer based on the retrieved documents 
        and chat history.

        Parameters:
        - question (str): The input to ask the agent.
        - session_id (str): The ID of the session to use for chat history.

        Returns:
        - str: The generated answer to the input.
        """

        config = {}
        if self.use_chat_history:
            if session_id is None:
                raise ValueError("Session ID is required when using chat history.")
            config = {"configurable": {"thread_id": session_id}}
        
        output = self.rag_chain.invoke({"input" : question}, config=config)
        
        self.logger.info("Answer: %s", output["answer"])
        
        return output

    def get_chat_history(self, session_id: str, pretty_print=True):
        chat_history = self.rag_chain.get_state({"configurable": {"thread_id": session_id}}).values["chat_history"]
        if pretty_print:
            for message in chat_history:
                message.pretty_print()
        return chat_history
