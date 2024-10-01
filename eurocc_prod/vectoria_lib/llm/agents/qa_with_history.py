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

from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph

from vectoria_lib.db_management.retriever.faiss_retriever import Retriever
from vectoria_lib.llm.parser import CustomResponseParser
from vectoria_lib.llm.helpers import format_docs
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
        
        #self.prompt = ChatPromptTemplate.from_template(get_prompt())            # PROMPT
        self.langchain_retriever = rag_retriever.as_langchain_retriever()        # RETRIEVER
        self.langchain_inference_engine = inference_engine.as_langchain_llm()    # LLM

        history_aware_retriever = self.get_history_aware_retriever()
        question_answer_chain = self.get_question_answer_chain()

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


        ### Statefully manage chat history ###
        # We define a dict representing the state of the application.
        # This state has the same input and output keys as `rag_chain`.
        class State(TypedDict):
            question: str
            chat_history: Annotated[Sequence[BaseMessage], add_messages]
            context: str
            answer: str


        # We then define a simple node that runs the `rag_chain`.
        # The `return` values of the node update the graph state, so here we just
        # update the chat history with the input message and response.
        def call_model(state: State):
            response = self.rag_chain.invoke(state)
            return {
                "chat_history": [
                    HumanMessage(state["question"]),
                    AIMessage(response["answer"]),
                ],
                "context": response["context"],
                "answer": response["answer"],
            }

        # Our graph consists only of one node:
        workflow = StateGraph(state_schema=State)
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)

        # Finally, we compile the graph with a checkpointer object.
        # This persists the state, in this case in memory.
        memory = MemorySaver()
        #app = workflow.compile(checkpointer=memory)
        self.qa_chain_with_memory = workflow.compile(checkpointer=memory)

    # --------------------------------------------------------------------------------

    def get_question_answer_chain(self):
        system_prompt = (
            "Sei un esperto di procedure aziendali. " 
            "Utilizza le seguenti porzioni di testo per rispondere alla domanda. "
            "Se non sai rispondere alla domanda, rispondi che non lo sai. "
            "Se la risposta non e' contenuta nel contesto, rispondi che non e' possibile rispondere dato il contesto. "
            "Utlizza tre frasi al massimo e mantieni la risposta precisa e breve."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )

        # TODO: we need to merge these two!!!!!!!!!!!!!!
        question_answer_chain = create_stuff_documents_chain(self.langchain_inference_engine, qa_prompt)
        # 2: question_answer_chain = (
        #     {"context": self.langchain_retriever | format_docs, "question": RunnablePassthrough()}
        #     | self.prompt
        #     | self.langchain_inference_engine
        #     | CustomResponseParser()
        # )
        # Merged version:
        # question_answer_chain = (
        #     { 
        #         "context": self.langchain_retriever | format_docs, # Retrieve and format the documents
        #         "question": RunnablePassthrough() # Step to pass through the question as is
        #     }
        #     | qa_prompt  # Use the qa_prompt template here to structure the input
        #     | self.langchain_inference_engine  # Run the inference engine on the formatted input
        #     # TODO: modify parser to be compliant with new prompt
        #     #| CustomResponseParser()  # Custom step to parse the engine's output
        # )

        return question_answer_chain
    
    # --------------------------------------------------------------------------------

    def get_history_aware_retriever(self):
        contextualize_q_system_prompt = (
            "Data la chat history e l'ultima domanda dell'utente "
            "che potrebbe fare riferimento al contesto contenuto nella chat history, "
            "formula una domanda indipendente e autocontenuta che può essere compresa "
            "senza la chat history. NON rispondere alla domanda, "
            "limitati a riformularla se necessario oppure restituiscila così com'è."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                # FIXME: this is a big problem damn
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.langchain_inference_engine,
            self.langchain_retriever,
            contextualize_q_prompt
        )

        return history_aware_retriever

    # --------------------------------------------------------------------------------

    def ask(self, question) -> str:
        """
        Ask the QAAgent a question and get an answer based on the retrieved documents 
        and chat history.

        Parameters:
        - question (str): The question to ask the agent.

        Returns:
        - str: The generated answer to the question.
        """

        config = {"configurable": {"thread_id": "abc123"}}

        # QUERY 1: defined in sbatch file ("Which are the energy and policy considerations for deep learning in NLP?")
        output = self.qa_chain_with_memory.invoke(
            {"question" : question},
            config = config)
        self.logger.info("\n\n=================> Answer:\n%s", output)

        # QUERY 2: follow up question that refers to the previous
        question2 = "How are they related?"
        output = self.qa_chain_with_memory.invoke(
            {"question" : question2},
            config = config)
        self.logger.info("\n\n=================> Answer:\n%s", output)
        return output
