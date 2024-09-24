from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.common.paths import TEST_DIR

def test_faiss_retriever():

    vector_store = FaissVectorStore.load_from_pickle(
        TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    
    retriever = FaissRetriever(search_kwargs={"k": 4})
    retriever.set_retriever(vector_store.as_retriever())

    query = "Who are the actors?"
    docs = retriever.get_docs(query)
    assert len(docs) == 4
