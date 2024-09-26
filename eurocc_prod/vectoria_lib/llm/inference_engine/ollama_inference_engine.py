from langchain_core.language_models.llms import BaseLLM
from langchain_community.chat_models import ChatOllama

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class OllamaInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Ollama.
    """
    def __init__(self, args: dict):
        super().__init__(args)
        
    def as_langchain_llm(self) -> BaseLLM:
        return ChatOllama(model = self.args["model_name"])