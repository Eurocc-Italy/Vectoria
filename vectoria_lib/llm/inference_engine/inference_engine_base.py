from abc import ABC, abstractmethod
from langchain_core.language_models.llms import BaseLLM

class InferenceEngineBase(ABC):
    def __init__(self, args: dict):
        self.args = args

    @abstractmethod
    def as_langchain_llm(self) -> BaseLLM:
        """
        Return the inference engine as a LangChain LLM.
        """
        pass