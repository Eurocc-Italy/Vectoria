import os
from vectoria_lib.common.config import Config
from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR

def test_agent_builder():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 1
    assert agent.chain.last is not None

def test_agent_builder_no_retriever():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    config.set("retriever", "enabled", False)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 0
    assert agent.chain.last is not None


def test_agent_builder_with_reranker():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    config.set("reranker", "enabled", True)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 2
    assert agent.chain.last is not None

def test_agent_builder_with_reranker_with_full_paragraphs_retrieval():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    config.set("reranker", "enabled", True)
    config.set("full_paragraphs_retrieval", "enabled", True)
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 2
    assert agent.chain.last is not None
