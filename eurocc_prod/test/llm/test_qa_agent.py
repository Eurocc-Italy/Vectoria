from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

import pytest
import torch

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    "inference_config", 
    [
        dict(
            name='ollama',
            model_name='llama3.2:1b'
        ),
        dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=100
        ),
        dict(
            name='openai',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            url="http://localhost:8899/v1",
            api_key="abcd"
        )
    ]
)
def test_qa_agent_ollama(inference_config):
    config = Config()
    config.set("inference_engine", inference_config)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl",
    )

    answer = agent.ask("What is the name of the movie?")

    assert "matrix" in answer.lower()
