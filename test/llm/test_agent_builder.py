from vectoria_lib.llm.agent_builder import AgentBuilder

def test_agent_builder(config, index_test_folder):
    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 1
    assert agent.chain.last is not None

def test_agent_builder_no_retriever(config, index_test_folder):
    config.set("retriever", "enabled", False)
    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 0
    assert agent.chain.last is not None


def test_agent_builder_with_reranker(config, index_test_folder):
    config.set("reranker", "enabled", True)
    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 2
    assert agent.chain.last is not None

def test_agent_builder_with_reranker_with_full_paragraphs_retriever(config, index_test_folder):
    config.set("reranker", "enabled", True)
    config.set("full_paragraphs_retriever", "enabled", True)
    agent = AgentBuilder.build_qa_agent(
        index_path=index_test_folder
    )
    assert agent is not None
    assert agent.chain is not None
    assert agent.chain.first is not None
    assert len(agent.chain.middle) == 3
    assert agent.chain.last is not None
