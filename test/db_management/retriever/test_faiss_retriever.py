import pytest 

from vectoria_lib.db_management.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.db_management.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.common.paths import TEST_DIR

@pytest.mark.parametrize("k",[1,2,3])
def test_faiss_retriever(k):

    vector_store = FaissVectorStore.load_from_pickle(
        TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    
    retriever = FaissRetriever(vector_store, search_type="mmr", search_kwargs={"k": k, "fetch_k": k, "lambda_mult": 0.5})

    query = "Who are the actors?"
    docs = retriever.get_docs(query)
    assert len(docs) == k
