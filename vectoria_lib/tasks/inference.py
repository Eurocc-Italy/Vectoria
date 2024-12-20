#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from vectoria_lib.llm.agent_builder import AgentBuilder

def inference(
    **kwargs: dict
):
    qa_agent = AgentBuilder.build_qa_agent(
        index_path=kwargs["index_path"]
    )
    qa_agent.inference(kwargs)
