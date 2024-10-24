import pytest
import torch

from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config
from langchain.docstore.document import Document

import pytest
import torch

@pytest.mark.slow
@pytest.mark.parametrize(
    "inference_config", 
    [   
        pytest.param(
            dict(
                name='ollama',
                model_name='llama3.2:1b'
            ),
            marks=pytest.mark.skipif(
                Config()
                .load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
                .get("inference_engine")["name"] != "ollama", 
                reason="ollama has not been configured"
            )
        ),
        pytest.param(
            dict(
                name='huggingface',
                model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                device="cuda",
                load_in_8bit=True,
                max_new_tokens=100,
                trust_remote_code=True,
                device_map=None
            ),
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
        ),
        pytest.param(
            dict(
                name='openai',
                model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
                url="http://localhost:8899/v1",
                api_key="abcd"
            ),
            marks=pytest.mark.skipif(
                Config()
                .load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
                .get("inference_engine")["name"] != "openai", 
                reason="openai has not been configured"
            )
        )
    ]
)
def test_qa_agent_engines(inference_config):
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    config.set("chat_history", False)
    config.set("retriever_top_k", 1)
    config.set("inference_engine", inference_config)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl",
    )

    answer = agent.ask("What is the name of the movie?")
    assert "answer" in answer
    assert "context" in answer
    assert "input" in answer

    assert len(answer["answer"]) > 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_with_history():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    inference_config = dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=200,
            trust_remote_code=False,
            device_map=None
        )   
    config.set("inference_engine", inference_config)
    config.set("documents_format", "pdf")
    config.set("chat_history", True)
    config.set("retriever_top_k", 1)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_airxv_papers.pkl"
    )

    answer = agent.ask(
        "Which are the energy and policy considerations for deep learning in NLP?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=False)
    assert len(chat_history) == 2

    answer = agent.ask(
        "How these two are related?",
        session_id="test_session"
    )
    chat_history = agent.get_chat_history("test_session", pretty_print=False)
    assert len(chat_history) == 4

    answer = agent.ask(
        "What did I ask before?",
        session_id="test_session_2"
    )
    chat_history = agent.get_chat_history("test_session_2", pretty_print=False)
    assert len(chat_history) == 2

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_without_history():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    inference_config = dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=200,
            trust_remote_code=False,
            device_map=None
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
    print(result)
    assert "answer" in result



@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_qa_agent_with_custom_context():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    inference_config = dict(
            name='huggingface',
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            device="cuda",
            load_in_8bit=True,
            max_new_tokens=200,
            trust_remote_code=False,
            device_map=None
        )   
    config.set("inference_engine", inference_config)
    config.set("documents_format", "pdf")
    config.set("chat_history", False)

    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_airxv_papers.pkl"
    )
    context = [
        Document("Deep learning in Natural Language Processing (NLP) is resource-intensive, and the energy and policy considerations are becoming increasingly important as models grow in size and complexity."),
        Document("Energy consumption: Deep learning models, especially large NLP models, require significant computational resources, leading to high energy consumption during both training and inference."),
        Document("Environmental impact: The energy-intensive nature of deep learning can contribute to increased carbon emissions, raising concerns about sustainability in AI development."),
        Document("Model efficiency: Optimizing model architectures, using more energy-efficient hardware, and employing techniques like model pruning or quantization can reduce energy demands without sacrificing performance."),
        Document("Policy frameworks: Governments and organizations are exploring regulations to ensure responsible AI practices, focusing on transparency, sustainability, and the ethical implications of energy use in AI technologies.")
    ]
    
    result = agent.ask_with_custom_context(
        "Which are the energy and policy considerations for deep learning in NLP?",
        context
    )
    print(result)
    assert isinstance(result, str)

