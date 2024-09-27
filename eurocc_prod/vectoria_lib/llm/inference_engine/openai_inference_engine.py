from langchain_core.language_models.llms import BaseLLM
from langchain_openai import OpenAI

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class OpenAIInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Ollama.
    """
    def __init__(self, args: dict):
        super().__init__(args)
        
    def as_langchain_llm(self) -> BaseLLM:
        return OpenAI(
            #temperature=self.args("temperature"),
            #max_retries=self.args("max_retries"),
            model = self.args["model_name"],
            base_url=self.args["url"],
            api_key=self.args["api_key"]
        )