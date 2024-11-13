#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from abc import ABC, abstractmethod
from langchain_core.language_models.llms import BaseLanguageModel
class InferenceEngineBase(ABC):
    def __init__(self, args: dict):
        self.args = args
        self.name = args["name"]
        self.model_name = args["model_name"]

    @abstractmethod
    def as_langchain_llm(self) -> BaseLanguageModel:
        """
        Return the inference engine as a LangChain LLM.
        """
        pass