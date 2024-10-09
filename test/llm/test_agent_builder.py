from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.common.paths import TEST_DIR

def test_agent_builder():
    agent = AgentBuilder.build_qa_agent(
        faiss_index_path=TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index_the_matrix.pkl"
    )
    assert agent is not None
