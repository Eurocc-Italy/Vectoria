#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings

class OpenAIInferenceEngine(InferenceEngineBase):

    def __init__(self, args: dict):
        super().__init__(args)

    def as_langchain_completion_model(self) -> BaseLanguageModel:
        return OpenAI(**self.args)
    
    def as_langchain_chat_model(self) -> BaseLanguageModel:
        return ChatOpenAI(**self.args)
    
    def as_langchain_embeddings_model(self) -> Embeddings:
        return OpenAIEmbeddings(**self.args)