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
def test_qa_agent_engines(inference_config):
    config = Config()
    config.set("inference_engine", inference_config)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl",
    )

    answer = agent.ask("What is the name of the movie?")

    assert "matrix" in answer.lower()


def test_qa_agent_with_history():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    inference_config = dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=500
        )   
    config.set("inference_engine", inference_config)
    config.set("documents_format", "pdf")
    config.set("chat_history", True)
    
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_airxv_papers.pkl"
    )

    answer = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=True)
    assert len(chat_history) == 2
    breakpoint()

    answer = agent.ask(
        "How these two are related?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=True)
    assert len(chat_history) == 4
    breakpoint()

    answer = agent.ask(
        "What did I ask before?",
        session_id="test_session_2"
    )
    chat_history = agent.get_chat_history("test_session_2", pretty_print=True)
    breakpoint()
    assert len(chat_history) == 2


def test_qa_agent_without_history():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    inference_config = dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=200
        )   
    config.set("inference_engine", inference_config)
    config.set("documents_format", "pdf")
    config.set("chat_history", False)
    config.set("retriever_top_k", 1) 

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_airxv_papers.pkl"
    )

    result = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?"
    )
    breakpoint()
    assert "answer" in result

