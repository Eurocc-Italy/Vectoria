import pytest 

from vectoria_lib.rag.retriever.faiss_retriever import FaissRetriever
from vectoria_lib.rag.vector_store.faiss_vector_store import FaissVectorStore

@pytest.mark.parametrize("k",[1,2,3])
def test_faiss_retriever(k, config, data_dir):
    
    vector_store = FaissVectorStore(
        model_name = config.get("vector_store", "model_name"),
        device = config.get("vector_store", "device"),
        normalize_embeddings = config.get("vector_store", "normalize_embeddings")
    ).load_from_disk(
        data_dir / "index" / "BAAI__bge-m3_faiss_index_the_matrix"
    )
    retriever = FaissRetriever(
        search_type = config.get("retriever", "search_type"),
        k = k,
        fetch_k = k, 
        lambda_mult = config.get("retriever", "lambda_mult")
    ).from_vector_store(
        vector_store
    )

    query = "Who are the actors of the movie?"

    docs = retriever.as_langchain_retriever().invoke(query)
    assert len(docs) == k
    