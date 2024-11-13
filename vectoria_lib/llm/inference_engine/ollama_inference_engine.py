#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain_community.chat_models import ChatOllama

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from langchain_core.language_models.llms import BaseLanguageModel

class OllamaInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Ollama.
    """
    def __init__(self, args: dict):
        super().__init__(args)
        
    def as_langchain_llm(self) -> BaseLanguageModel:
        return ChatOllama(model = self.args["model_name"])