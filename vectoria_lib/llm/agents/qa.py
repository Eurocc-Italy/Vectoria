#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging, datetime, time
from pathlib import Path

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langsmith import traceable

from vectoria_lib.llm.agents.stateful_workflow import StatefulWorkflow
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.llm.parser import CustomResponseParser
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
        
        if rag_retriever is not None:
            retriever = rag_retriever.as_langchain_retriever()
        else:
            self.logger.warning("No retriever provided, the RAG retriever will be not be initialized")
            retriever = None
        
        inference_engine = inference_engine.as_langchain_llm()

        prompt_builder = PromptBuilder(system_prompts_lang)

        self.use_chat_history = chat_history
        self.rag_chain = None
        self.oracle_chain = None
        if self.use_chat_history is True and retriever is not None:

            self.logger.info("Creating QA agent with the RAG retriever and chat history")
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

        self.logger.info("Creating QA agent with the oracle retriever")
        self.oracle_chain = create_stuff_documents_chain(
            inference_engine,
            prompt_builder.get_qa_prompt(),
            output_parser=CustomResponseParser()
        )

        if retriever is not None and self.use_chat_history is False:
            self.logger.info("Creating QA agent with the RAG retriever")
            self.rag_chain = create_retrieval_chain(
                retriever,
                self.oracle_chain
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
        - dict: The generated answer to the input.
        """

        config = {}
        if self.use_chat_history:
            if session_id is None:
                raise ValueError("Session ID is required when using chat history.")
            config = {"configurable": {"thread_id": session_id}}
        
        output = self.rag_chain.invoke({"input" : question}, config=config)
        
        self.logger.debug("\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question)
        self.logger.info( "\n------%s------- > Answer: %s",   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), output["answer"])

        return output

    def ask_with_custom_context(self, question: str, context: list[Document]) -> str:
        if self.use_chat_history:
            self.logger.warning("This method is not supported when using chat history.")
            return None
        return self.oracle_chain.invoke({"input" : question, "context" : context})

    def get_chat_history(self, session_id: str, pretty_print=True):
        chat_history = self.rag_chain.get_state({"configurable": {"thread_id": session_id}}).values["chat_history"]
        if pretty_print:
            for message in chat_history:
                message.pretty_print()
        return chat_history

    def inference(self, test_set_path: str, output_dir: str):
        """
        {
            "question": [
                "Da quali fasi si compone l'attività di gestione delle richieste di acquisto?",
                "Qual'è l'obbiettivo dell'attività di gestione delle richieste di acquisto?"
            ],
            "answer": [
                "answer1",
            ],
            "contexts": [
                ["context1", "context2", "context3"],
            ],
            "ground_truth": [
                "L'attività di gestione delle richieste di acquisto è composta dalle seguenti fasi: Emissione della RdA (manuale / MRP). Approvazione / Autorizzazione della RdA. Presa in carico della RdA da parte del Procurement/LGS.",
                "L'obbiettivo dell'attività di gestione delle richieste di acquisto è Garantire che il fabbisogno espresso da un Ente Aziendale sia correttamente trasformato in una richiesta di acquisto gestibile dall'Unità Operativa Procurement."
            ]  
        }    
        """
        import json
        import numpy as np
        with open(test_set_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        times = []
        output = data.copy()
        output["contexts"] = []
        output["answer"] = []
        for q in data["question"]:
            start_time = time.perf_counter()
            result = self.ask(q)
            took = time.perf_counter() - start_time
            self.logger.info("Time taken to answer question: %.2f seconds", took)
            times.append(took)
            contexts = result["context"]
            answer = result["answer"]
            output["contexts"].append([c.page_content for c in contexts])
            output["answer"].append(answer)
        
        self.logger.info("Mean time and std taken to answer questions: %.2f seconds, %.2f seconds", np.mean(times), np.std(times))

        Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_file = Path(output_dir) / f"{Path(test_set_path).stem}_with_answers_and_contexts.json"
        start_time = time.perf_counter()
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(output, file, indent=4, ensure_ascii=False)

        self.logger.info("Annotated test set saved to %s and took %.2f seconds", output_file, time.perf_counter() - start_time)

