import pytest
from components.rerankers.openai_vllm import OpenAIvLLMReranker

def test_vllm_reranker(openai_server_status_fn):
    inference_config = {
        "name": "vllm",
        "model_name": "BAAI/bge-reranker-v2-m3",
        "openai_api_base": "http://localhost:8002/v1",
        "openai_api_key": "EMPTY",
        "top_n": 1
    }
    if not openai_server_status_fn(inference_config):
        pytest.skip("OpenAI server is not running")

    reranker = OpenAIvLLMReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        host="localhost",
        port=8002,
        top_n=1
    )

    scores = reranker.score([
        ("What is the capital of Brazil?", "The capital of Brazil is Brasilia"),
        ("What is the capital of Brazil?", "The capital of France is Paris")
    ])

    assert scores[0] > 0.9
    assert scores[1] < 0.1
