from eurocc_v1.lib.rag.retriever import Retriever

class FaissRetriever(Retriever):

    """
    This class is a wrapper around the FAISS library to provide a retriever for the RAG model.

    Example usage:
        vector_store = FaiseVectorStore().make_index(docs)
        retriever = FaissRetriever(search_kwargs={"k": 5})
        retriever.set_retriever(vector_store.as_retriever())
            
        query = "Qual'è l'argomento principale trattato nella procedura PRO012-P?"
        docs = retriever.get_docs(query)

        for id, doc in enumerate(docs):
            print(f"\n+++++++++++ CHUNK # {id} +++++++++++")
            print(doc.page_content)
            print("\n############################################################")

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_retriever(self, retriever):
        self.retriever = retriever

    def get_docs(self, query: str):
        return self.retriever.get_relevant_documents(query)

    def as_langchain_retriever(self):
        return self.retriever



