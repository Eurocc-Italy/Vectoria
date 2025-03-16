import pytest

from vectoria_lib.common.config import Config
from vectoria_lib.chains.retrieval import get_retrieval_chain
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory

def test_retrieval_chain(config, index_test_folder):
    _ = VectorStoreFactory.build_vector_store(
        **config.get("vector_store")
    ).load_index(index_test_folder)

    chain = get_retrieval_chain()

    result = chain.invoke({"input": "What is the capital of France?"})
    assert result is not None
    assert len(result) == 5


def test_retrieval_chain_with_hugging_face_reranker(config, index_test_folder):
    _ = VectorStoreFactory.build_vector_store(
        **config.get("vector_store")
    ).load_index(index_test_folder)

    config.set("retriever", "enable_rerank", True)
    config.set("retriever", "rerank_k", 2)
    inference_engine = {
        "name": "huggingface",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "device": "cuda",
    }
    config.set("retriever", "inference_engine", inference_engine)

    chain = get_retrieval_chain()

    result = chain.invoke({"input": "What is the capital of France?"})
    assert result is not None
    assert len(result) == 2


def test_retrieval_chain_with_openai_reranker(config, index_test_folder, openai_server_status_fn):
    inference_config = {
        "name": "openai",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "openai_api_base": "http://localhost:8000", # /score is not part of the standard openai api so /v1 is not added
        "openai_api_key": "EMPTY"
    }
    if not openai_server_status_fn(inference_config):
        pytest.skip("OpenAI server is not running")

    _ = VectorStoreFactory.build_vector_store(
        **config.get("vector_store")
    ).load_index(index_test_folder)

    config.set("retriever", "enable_rerank", True)
    config.set("retriever", "rerank_k", 2)
    config.set("retriever", "inference_engine", inference_config)

    chain = get_retrieval_chain()

    result = chain.invoke({"input": "What is the capital of France?"})
    assert result is not None
    assert len(result) == 2


