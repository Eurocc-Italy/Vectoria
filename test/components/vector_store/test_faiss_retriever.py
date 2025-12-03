import pytest 

from vectoria_lib.components.vector_store.faiss_vector_store import FaissVectorStore
from vectoria_lib.components.llm.llm_factory import LLMFactory
@pytest.mark.parametrize("k",[1,2,3])
def test_faiss_retriever(k, config, index_test_folder):
    
    embedder = LLMFactory.build_llm(config.get("vector_store", "inference_engine")).as_langchain_embeddings_model()

    vector_store = FaissVectorStore(
        embedder_model = embedder,
        index_path = None
    ).load_index(
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
    