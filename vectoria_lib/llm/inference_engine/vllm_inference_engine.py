#
# VECTORIA
#
# @authors : Andrea Proia, Leonardo Baroncelli
#

from langchain_openai import OpenAI

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from langchain_core.language_models.llms import BaseLanguageModel

class VLLMInferenceEngine(InferenceEngineBase):
    """
    Wrapper on OpenAI.
    """
    def __init__(self, args: dict):
        super().__init__(args)
    
    # TODO: OpenAI or ChatOpenAI?
    def as_langchain_llm(self) -> BaseLanguageModel:
        return OpenAI(
            model = self.args["model_name"],
            base_url = self.args["url"],
            api_key = self.args["api_key"]
        )