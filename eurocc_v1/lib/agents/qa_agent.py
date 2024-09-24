import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain

from eurocc_v1.lib.rag.retriever import Retriever
from eurocc_v1.lib.agents.output_parsers.custom import CustomResponseParser

class QAAgent():

    def __init__(
        self,
        rag_retriever: Retriever,
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_new_tokens = 150,
        load_in_8bit = True,
        device = "cuda"
    ):
        """
            max_new_tokens: https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
        """
        self.logger = logging.getLogger('ecclogger')

        self.model_name = model_name
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            load_in_8bit = load_in_8bit
        )
        if load_in_8bit:
            device = None
        
        pipe = pipeline(
            "text-generation", 
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = max_new_tokens,
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
            return "\n\n".join(doc.page_content for doc in docs)
            
        langchain_retriever = rag_retriever.as_langchain_retriever()
        
        self.chain = (
            {"context": langchain_retriever | format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.pipe
            | CustomResponseParser()
        )

    #question = "Qual'è l'argomento principale trattato nella procedura PRO012-P?"

    def ask(self, question):
        output = self.chain.invoke(question)
        self.logger.info(f"\n\n\nQAAgent: {output}")
        return output

