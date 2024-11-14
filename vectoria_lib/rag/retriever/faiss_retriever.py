#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
from vectoria_lib.rag.retriever.retriever_base import RetrieverBase
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore
class FaissRetriever(RetrieverBase):

    def __init__(self, **kwargs):
        """
        Constructor that stores the search_type and search_kwargs
        Parameters:
        - Keyword arguments: search_type: str (e.g. "mmr"), search_kwargs: dict (other configs: "k", "fetch_k", "lambda_mult")
        """
        super().__init__(**kwargs)
        self.search_type = kwargs["search_type"]
        self.search_kwargs = {
            "k": kwargs["k"],
            "fetch_k": kwargs["fetch_k"],
            "lambda_mult": kwargs["lambda_mult"]
        }

    def from_vector_store(self, vector_store: FaissVectorStore):
        self.logger.info("Creating retriever from vector store with kwargs: %s" % self.search_kwargs)
        self.wrapped_retriever = vector_store.as_retriever(
            search_type=self.search_type, 
            search_kwargs=self.search_kwargs
        )
        return self

    def as_langchain_retriever(self):
        return self.wrapped_retriever
