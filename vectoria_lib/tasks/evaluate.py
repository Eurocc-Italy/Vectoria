#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.llm.agent_evaluator import AgentEvaluator

def evaluate(
    **kwargs: dict
):
    qa_agent = AgentBuilder.build_qa_agent(**kwargs)
    evaluator = AgentEvaluator(qa_agent)

    evaluator.evaluate(kwargs["test_set_path"])
    evaluator.dump()
    #evaluator.ragas_eval()
            
    return None