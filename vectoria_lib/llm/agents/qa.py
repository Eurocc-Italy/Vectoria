#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import json, logging, datetime, time
from pathlib import Path
from langsmith import traceable
from langchain.docstore.document import Document

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
    def ask(self, question: str, context: list[Document] = None, session_id: str = None, config: dict = {}) -> str:
        """
        Ask the QAAgent a input and get an answer based on the retrieved documents 
        and chat history.

        Parameters:
        - question (str): The input to ask the agent.
        - session_id (str): The ID of the session to use for chat history.

        Returns:
        - dict: The generated answer to the input.
        """

        #if self.use_chat_history:
        #    if session_id is None:
        #        raise ValueError("Session ID is required when using chat history.")
        #    config = {"configurable": {"thread_id": session_id}}
        inputs = {"input" : question}

        if context is not None:
            inputs["docs"] = context

        results = self.chain.invoke(inputs, config)
        
        self.logger.debug("\n------%s------- > Question: %s", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question)
        self.logger.info( "\n------%s------- > Answer: %s",   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), results["answer"])

        return results


    # def get_chat_history(self, session_id: str, pretty_print=True):
    #     chat_history = self.chain.get_state({"configurable": {"thread_id": session_id}}).values["chat_history"]
    #     if pretty_print:
    #         for message in chat_history:
    #             message.pretty_print()
    #     return chat_history
    def _get_correct_context_key(self, result: dict):
        if "full_paragraphs_docs" in result:
            return "full_paragraphs_docs"
        elif "reranked_docs" in result:
            return "reranked_docs"
        else:
            return "docs"

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
        
        Path(output_dir).mkdir(exist_ok=True, parents=True)

        times = []
        results = data.copy()
        results["contexts"] = []
        results["answer"] = []
        for q in data["question"]:

            start_time = time.perf_counter()
            try:
                result = self.ask(q)
            except Exception as e:
                self.logger.error("Error answering question: %s", e)
                self._write_inference_results(results, output_dir, Path(test_set_path).stem)
                return 
            took = time.perf_counter() - start_time
            self.logger.info("Time taken to answer question: %.2f seconds", took)
            times.append(took)

            contexts_key = self._get_correct_context_key(result)
            contexts = result[contexts_key]
            answer = result["answer"]
            if contexts_key not in results:
                results[contexts_key] = []
            results[contexts_key].append([c.page_content for c in contexts])
            results["answer"].append(answer)
        
        self.logger.info("Mean time and std taken to answer questions: %.2f seconds, %.2f seconds", np.mean(times), np.std(times))

        self._write_inference_results(results, output_dir, Path(test_set_path).stem)
    
    def _write_inference_results(self, results, output_dir, output_name):
        output_file = Path(output_dir) / f"{output_name}_with_answers_and_contexts.json"
        start_time = time.perf_counter()
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

        self.logger.info("Annotated test set saved to %s and took %.2f seconds", output_file, time.perf_counter() - start_time)

