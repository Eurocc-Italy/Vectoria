import pytest 

from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore

@pytest.mark.parametrize("k",[1,2,3])
def test_faiss_retriever(k, config, index_test_folder):
    
    vector_store = FaissVectorStore(
        model_name = config.get("vector_store", "model_name"),
        device = config.get("vector_store", "device"),
        normalize_embeddings = config.get("vector_store", "normalize_embeddings")
    ).load_from_disk(
        index_test_folder
    )
    retriever = vector_store.as_retriever(
        search_config = {
            "search_type": config.get("retriever", "search_type"),
            "k": k,
            "fetch_k": k, 
            "lambda_mult": config.get("retriever", "lambda_mult")
        }
    )

    query = "Who are the actors of the movie?"

    docs = retriever.invoke(query)
    assert len(docs) == k
    