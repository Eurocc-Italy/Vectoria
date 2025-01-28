#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
import numpy as np
from pathlib import Path
import json, logging, datetime, time
from langsmith import traceable
from langchain.docstore.document import Document
from vectoria_lib.common.config import Config

class QAAgent:
    """
    A Question Answering (QA) Agent that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The agent can generate answers using a language model and 
    return a response based on the retrieved documents.
    """

    def __init__(
        self,
        chain
    ):
        """
        Initialize the QAAgent object with the corresponding chains

        Parameters:

        """
        self.logger = logging.getLogger('llm')
        self.chain = chain



    # --------------------------------------------------------------------------------
    @traceable
    def ask(self, question: str, context: list[Document] = None) -> dict:
        """
        Ask the QAAgent a input and get an answer based on the retrieved documents 
        and chat history.

        Parameters:
        - question (str): The input to ask the agent.
        - session_id (str): The ID of the session to use for chat history.

        Returns:
        - dict: The generated answer to the input.
        """
        inputs = {"input" : question}

        if context is not None:
            inputs["docs"] = context

        results = self.chain.invoke(inputs)
        
        self.logger.debug("\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question)
        self.logger.info( "\n------%s------- > Answer: %s",   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), results["answer"])

        return results

    def _get_correct_context_key(self):
        config = Config()
        
        if config.get("full_paragraphs_retriever", "enabled"):
            return "full_paragraphs_docs"
        
        elif config.get("reranker", "enabled"):
            return "reranked_docs"
        
        else:
            return "docs"

    def _get_questions(self, kwargs):
        if kwargs.get("questions"):
            # Handle questions from CLI
            data = {"question": kwargs.get("questions")}
            self.logger.info("Received questions directly via CLI: %s", kwargs.get("questions"))
        else:
            # Handle questions from the test set JSON
            test_set_path = kwargs.get("test_set_path")
            if not test_set_path:
                raise ValueError("No questions provided. Use '--questions' or '--test-set-path'.")
            with open(test_set_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            self.logger.info("Questions are loaded from test set JSON: %s", test_set_path)
        return data

    def inference(self, kwargs):

        data = self._get_questions(kwargs)
        output_dir = kwargs.get("output_dir")

        times = []
        results = data.copy()
        results["answer"] = []
        retrieved_contexts_key = self._get_correct_context_key()
        results[retrieved_contexts_key] = []

        for q in data["question"]:
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

            contexts = result[retrieved_contexts_key]
            answer = result["answer"]

            results[retrieved_contexts_key].append(
                [{
                    "page_content": c.page_content,
                    "metadata": c.metadata
                } for c in contexts ]
            )
            results["answer"].append(answer)
        
        results["times"] = dict(
            mean = np.mean(times),
            std = np.std(times)
        )
        self.logger.info("Mean time and std taken to answer questions: %.2f seconds, %.2f seconds", results["times"]["mean"], results["times"]["std"])
        self._write_inference_results(results, output_dir, f"inference_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")    

    
    def _write_inference_results(self, results, output_dir, output_name):
        Path(output_dir).mkdir(exist_ok=True, parents=True)
        output_file = Path(output_dir) / f"{output_name}.json"
        start_time = time.perf_counter()
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        self.logger.info("Annotated test set saved to %s and took %.2f seconds", output_file, time.perf_counter() - start_time)

