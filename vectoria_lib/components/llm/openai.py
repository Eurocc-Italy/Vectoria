#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

from vectoria_lib.components.llm.llm_base import LLMBase
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from vectoria_lib.components.rerankers.openai_vllm import OpenAIvLLMReranker

class OpenAILLM(LLMBase):

    def __init__(self, args: dict):
        super().__init__(args.copy())
        self.args.pop("name")

    def as_langchain_completion_model(self) -> BaseLanguageModel:
        return OpenAI(**self.args)
    
    def as_langchain_chat_model(self) -> BaseLanguageModel:
        return ChatOpenAI(**self.args)
    
    def as_langchain_embeddings_model(self) -> Embeddings:
        return OpenAIEmbeddings(**self.args)
    
    def as_langchain_reranker_model(self) -> BaseLanguageModel:
        return OpenAIvLLMReranker(**self.args)
