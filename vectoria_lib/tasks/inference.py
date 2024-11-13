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
        faiss_index_path=kwargs["faiss_index_path"]
    )
    qa_agent.inference(test_set_path=kwargs["test_set_path"], output_dir=kwargs["output_dir"])
