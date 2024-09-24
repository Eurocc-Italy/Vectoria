#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from vectoria_lib.db_management.retriever.retriever import Retriever

class FaissRetriever(Retriever):

    """
    Wrapper class around the FAISS library to provide a document retriever for the RAG (Retrieval-Augmented Generation) model.

    This class allows for setting a FAISS-based retriever and fetching relevant documents for a given query.
    
    Example usage:
        vector_store = FaiseVectorStore().make_index(docs)     # Create a FAISS index with the documents.
        retriever = FaissRetriever(search_kwargs={"k": 5})     # Initialize the FaissRetriever with a search limit of 5 results.
        retriever.set_retriever(vector_store.as_retriever())   # Set the FAISS retriever instance.

        query = "Qual'è l'argomento principale trattato nella procedura PRO012-P?"  # Example query in Italian.
        docs = retriever.get_docs(query)    # Retrieve the most relevant documents based on the query.

        for id, doc in enumerate(docs):
            print(f"\n+++++++++++ CHUNK # {id} +++++++++++")
            print(doc.page_content)
            print("\n############################################################")
    """

    def __init__(self, **kwargs):
        """
        Initialize the FaissRetriever object.

        """
        super().__init__(**kwargs)

    def set_retriever(self, retriever):
        """
        Set the FAISS retriever to be used for fetching documents.

        Parameters:
        - retriever: The FAISS retriever instance that will be used to retrieve documents.

        Returns:
        - None
        """
        self.retriever = retriever

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



