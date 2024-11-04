from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableLambda

class BaseReranker(ABC):
    # TODO: DEPRECATED
    def __init__(self, args: dict):
        self.args = args

    @abstractmethod
    def as_langchain_reranker(self) -> RunnableLambda:
        """
        Wraps rerank into a LangChain runnable
        """
        pass
