#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from abc import ABC, abstractmethod

class Retriever(ABC):
    """
    Abstract base class for document retrievers.

    """

    def __init__(self, **kwargs):
        """
        Initialize the Retriever object.

        Parameters:
        - kwargs: Optional keyword arguments for retriever configuration.
        """
        self.kwargs = kwargs
        self.retriever = None

    @abstractmethod
    def set_retriever(self, retriever):
        """
        Abstract method to set the retriever instance.

        Parameters:
        - retriever: The retriever instance to be set.
        
        Returns:
        - None

        """
        pass

    @abstractmethod
    def get_docs(self, query: str) -> list[object]:
        """
        Abstract method to retrieve relevant documents based on a query.

        Parameters:
        - query (str): The query string for which to fetch relevant documents.

        Returns:
        - list[object]: A list of documents (or document-like objects) relevant to the query.
        """
        pass