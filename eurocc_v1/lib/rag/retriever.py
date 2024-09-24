from abc import ABC, abstractmethod

class Retriever(ABC):

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.retriever = None

    @abstractmethod
    def set_retriever(self, retriever):
        pass

    @abstractmethod
    def get_docs(self, query: str) -> list[object]:
        pass