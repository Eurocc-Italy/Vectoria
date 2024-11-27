import logging
from abc import ABC, abstractmethod
from vectoria_lib.rag.vector_store.vectore_store_base import VectorStoreBase

class RetrieverBase(ABC):

    def __init__(self, **kwargs):
        self.logger = logging.getLogger('rag')
        self.kwargs = kwargs

        self.wrapped_retriever = None
        self.wrapped_vector_store = None

    @abstractmethod
    def from_vector_store(self, vector_store: VectorStoreBase):
        pass

    @abstractmethod
    def as_langchain_retriever(self):
        """
        Return the retriever in a format compatible with Langchain Runnables.
        This is the method/object that will be called by the Langchain framework to retrieve the chunks from the vector store.
        """
        pass