from vectoria_lib.common.config import Config
from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR

def test_agent_builder():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
    assert agent.rag_chain is not None
    assert agent.oracle_chain is not None

def test_agent_builder_no_index():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=None
    )
    assert agent.rag_chain is None
    assert agent.oracle_chain is not None