#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
from vectoria_lib.llm.agent_builder import AgentBuilder
from vectoria_lib.llm.agent_evaluator import AgentEvaluator
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.config import Config

def evaluate(
    **kwargs: dict
):

    evaluator = AgentEvaluator()

    config = Config()

    config.set("inference_engine", dict(
        name='openai',
        model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
        url="http://localhost:8899/v1",
        api_key="abcd"
    ))

    if "generate_answers" in kwargs and kwargs["generate_answers"]:
        data = evaluator.generate_answers(kwargs["test_set_path"], dump=True)
    else:
        data = evaluator.load_from_yaml(kwargs["test_set_path"])
    
    evaluator.ragas_eval(
        data,
        InferenceEngineBuilder.build_inference_engine(
            config.get("inference_engine")).as_langchain_llm()
    )