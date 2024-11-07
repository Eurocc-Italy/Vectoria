#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from langchain_community.vectorstores.faiss import FAISS

class FaissRetriever:

    def __init__(self, vector_store: FAISS, search_type: str, search_kwargs: dict):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs)

    def as_langchain_retriever(self):
        """
        Return the FAISS retriever in a format compatible with Langchain retrievers.

        Returns:
        - retriever: The current FAISS retriever instance.
        """
        return self.retriever

    def retrieve_paragraphs(self, chunks: List[Document]):
        self.retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"lambda_mult": 0.5})
        chunks = self.retriever.get_relevant_documents(chunks)


