#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain_community.vectorstores.faiss import FAISS

class FaissRetriever:

    def __init__(self, vector_store: FAISS, search_type: str, search_kwargs: dict):
        self.retriever = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)

    def get_docs(self, query: str):
        """
        Retrieve documents relevant to the provided query using the FAISS retriever.

        Parameters:
        - query (str): The query string for which to retrieve relevant documents.

        Returns:
        - list: A list of documents that are most relevant to the provided query.
        """
        return self.retriever.get_relevant_documents(query)

    def as_langchain_retriever(self):
        """
        Return the FAISS retriever in a format compatible with Langchain retrievers.

        Returns:
        - retriever: The current FAISS retriever instance.
        """
        return self.retriever



