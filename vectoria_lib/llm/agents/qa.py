#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging, datetime, time
from pathlib import Path
from langsmith import traceable

from vectoria_lib.llm.agents.stateful_workflow import StatefulWorkflow
from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.db_management.reranking.reranker_base import BaseReranker
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from langchain.docstore.document import Document
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate

class QAAgent:
    """
    A Question Answering (QA) Agent that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The agent can generate answers using a language model and 
    return a response based on the retrieved documents.
    """

    def __init__(
        self,
        oracle_chain,
        chain = None
    ):
        """
        Initialize the QAAgent object with the corresponding chains

        Parameters:

        """
        self.logger = logging.getLogger('llm')
        self.oracle_chain = oracle_chain
        self.chain = chain



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
        #if self.use_chat_history:
        #    if session_id is None:
        #        raise ValueError("Session ID is required when using chat history.")
        #    config = {"configurable": {"thread_id": session_id}}
        
        output = self.chain.invoke({"input" : question}, config=config)
        
        self.logger.debug("\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question)
        self.logger.info( "\n------%s------- > Answer: %s",   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), output["answer"])

        return output

    def ask_with_custom_context(self, question: str, context: list[Document]) -> str:
        #if self.use_chat_history:
        #    self.logger.warning("This method is not supported when using chat history.")
        #    return None
        return self.oracle_chain.invoke({"input" : question, "docs" : context})

    # def get_chat_history(self, session_id: str, pretty_print=True):
    #     chat_history = self.chain.get_state({"configurable": {"thread_id": session_id}}).values["chat_history"]
    #     if pretty_print:
    #         for message in chat_history:
    #             message.pretty_print()
    #     return chat_history

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

