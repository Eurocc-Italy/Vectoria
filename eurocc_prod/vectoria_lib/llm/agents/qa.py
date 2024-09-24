#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------

import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain

from vectoria_lib.common.config import Config
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.db_management.retriever.faiss_retriever import Retriever, FaissRetriever
from vectoria_lib.llm.parser import CustomResponseParser

class QAAgent():
    """
    A Question Answering (QA) Agent that leverages a Retrieval-Augmented Generation (RAG) retriever and a language model 
    to answer questions based on a provided context. The agent can generate answers using a language model and 
    return a response based on the retrieved documents.
    """

    def __init__(
        self,
        rag_retriever: Retriever
    ):
        """
            max_new_tokens: https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
        """"""
        Initialize the QAAgent object with the provided retriever and language model configuration.

        Parameters:
        - rag_retriever (Retriever): A retriever object to fetch relevant documents from a FAISS-based vector store.
        - model_name (str): The name of the pre-trained language model to be used for text generation. Default is "meta-llama/Meta-Llama-3.1-8B-Instruct".
        - max_new_tokens (int): The maximum number of tokens that the model can generate in response.
                                Reference: https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
        - load_in_8bit (bool): Whether to load the model in 8-bit precision to save memory. Default is True.
        - device (str): The device to run the model on, such as "cuda" or "cpu". Default is "cuda".
        """

        self.logger = logging.getLogger('llm')
        config = Config()

        self.model_name = config.get("llm_model")
        tokenizer = AutoTokenizer.from_pretrained(config.get("llm_model"))
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit = config.get("load_in_8bit")
        )
        if config.get("load_in_8bit"):
            device = None
        
        pipe = pipeline(
            "text-generation", 
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = config.get("max_new_tokens"),
            device = device
        )

        # https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/
        self.pipe = HuggingFacePipeline(pipeline=pipe)

        # Se la risposta non e' contenuta nel CONTESTO, rispondi che non lo sai.
        
        prompt = """Sei un esperto di procedure aziendali. Il tuo obiettivo è scrivere
        una RISPOSTA alla domanda che ti viene posta, dato il CONTESTO e la DOMANDA che seguono.
        Se la risposta non e' contenuta nel CONTESTO, rispondi che non lo sai.

        ======================CONTESTO=============================
        CONTESTO = {context}

        ======================DOMANDA=============================
        DOMANDA = {question}

        ======================RISPOSTA=============================
        RISPOSTA =
        """

        self.prompt = ChatPromptTemplate.from_template(prompt)

        def format_docs(docs):
            """
            Helper function to format the retrieved documents for use in the context.

            Parameters:
            - docs (list): A list of document objects containing content.

            Returns:
            - str: Formatted string with concatenated page content from the documents.
            """
            
            return "\n\n".join(doc.page_content for doc in docs)
            
        langchain_retriever = rag_retriever.as_langchain_retriever()
        
        self.chain = (
            {"context": langchain_retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.pipe
            | CustomResponseParser()
        )


    def ask(self, question):
        """
        Ask the QAAgent a question and get an answer based on the retrieved documents 
        and generated response.

        Parameters:
        - question (str): The question to ask the agent.

        Returns:
        - str: The generated answer to the question.
        """
        output = self.chain.invoke(question)
        self.logger.info("\n\n=================> Answer:\n%s", output)
        return output
