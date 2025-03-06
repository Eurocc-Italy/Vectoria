#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
import numpy as np
from pathlib import Path
import json, logging, datetime, time
from typing import List

from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough

from vectoria_lib.common.config import Config
from vectoria_lib.applications.chain_runner import ChainRunner
from vectoria_lib.chains.retrieval import get_retrieval_chain
from vectoria_lib.chains.reranking import get_reranking_chain
from vectoria_lib.chains.create_context import get_create_context_chain
from vectoria_lib.chains.context_enhancement import get_context_enhancement_chain
from vectoria_lib.chains.generation import get_generation_chain
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory

class QAApplication(ChainRunner):
    """
    A Question Answering (QA) application that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The app can generate answers using a language model and return a response 
    based on the retrieved documents.

    Class responsibilities:
        * Define the chain
        * Expose an API to answer questions
        * Run the chain
        * Handle the output of the chain
    """

    def __init__(
        self,
        index_path: Path      
    ):
        """
        Initialize the QAApplication object with the corresponding chains
        """
        self.logger = logging.getLogger('llm')

        config = Config()

        # Load the index from the disk
        _ = VectorStoreFactory.create_vector_store(**config.get("vector_store")).load_from_disk(index_path)

        chain = RunnablePassthrough() # The RunnablePassthrough makes the input key pass through all the next runnables to the final output dictionary

        if config.get("retriever", "enabled"):
            chain = chain.assign(docs=get_retrieval_chain())

        if config.get("reranker", "enabled"):
            chain = chain.assign(docs=get_reranking_chain())

        if config.get("full_paragraphs_retriever", "enabled"):
            chain = chain.assign(docs=get_context_enhancement_chain())

        chain = chain.assign(context=get_create_context_chain())

        chain = chain.assign(answer=get_generation_chain())

        super().__init__(chain, config.get("langfuse"))


    def ask(
            self, 
            question: str, 
            context: list[Document] = None
    ) -> dict:
        """
        Ask the QAApplication a input and get an answer based on the retrieved documents 
        and chat history.

        Parameters:
        - question (str): The input to ask the app.
        - session_id (str): The ID of the session to use for chat history.

        Returns:
        - dict: The generated answer to the input.
        """
        inputs = {"input" : question}

        if context is not None:
            inputs["docs"] = context

        results = self.invoke(inputs)

        self.logger.debug("\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question)
        self.logger.info( "\n------%s------- > Answer: %s",   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), results["answer"])

        return results

    def _get_questions(self, kwargs) -> List[str]:

        if not kwargs.get("test_set_path") and not kwargs.get("questions"):
            raise ValueError("No questions provided. Use '--questions' or '--test-set-path'.")

        if kwargs.get("questions"):
            # Handle questions from CLI
            self.logger.info("Received questions directly via CLI: %s", kwargs.get("questions"))
            return kwargs.get("questions")

        # Handle questions from the test set JSON
        if kwargs.get("test_set_path"):
            with open(kwargs.get("test_set_path"), 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.logger.info("Questions are loaded from test set JSON: %s", kwargs.get("test_set_path"))
            return data["question"]

    def inference(self, kwargs):

        questions = self._get_questions(kwargs)
        output_dir = kwargs.get("output_dir")

        times = []
        collected_results = {
            "question": questions,
            "contexts": [],
            "answer": [],
            "docs": []
        }

        for q in questions:

            self.logger.info("Answering question: %s", q)

            start_time = time.perf_counter()
            try:
                result = self.ask(q)
            except Exception as e:
                self.logger.error("Error answering question: %s", e)
                return
            
            took = time.perf_counter() - start_time
            self.logger.info("Time taken to answer question: %.2f seconds", took)
            times.append(took)


            # update the results
            collected_results["answer"].append(result["answer"])
            if "docs" in result:
                for c in result["docs"]:
                    collected_results["docs"].append({
                        "page_content": c.page_content,
                        "metadata": c.metadata
                    })
            collected_results["contexts"].append(result["context"])

        
        collected_results["times"] = dict(
            mean = np.mean(times),
            std = np.std(times)
        )
        self.logger.info("Mean time and std taken to answer questions: %.2f seconds, %.2f seconds", collected_results["times"]["mean"], collected_results["times"]["std"])
        self._write_inference_results(collected_results, output_dir, f"inference_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")    

    
    def _write_inference_results(self, results, output_dir, output_name):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_file = Path(output_dir) / f"{output_name}.json"
        start_time = time.perf_counter()
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        self.logger.info("Annotated test set saved to %s and took %.2f seconds", output_file, time.perf_counter() - start_time)

