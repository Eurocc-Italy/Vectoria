from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

import pytest
import torch

@pytest.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent():
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )

    assert agent.model_name == Config().get("llm_model")
    
    answer = agent.ask("What is the name of the movie?")

    assert "matrix" in answer.lower()
