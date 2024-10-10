#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
from vectoria_lib.llm.agent_evaluator import AgentEvaluator
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.config import Config

def load_yaml(path: str):
    import yaml
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data

def evaluate(
    **kwargs: dict
):

    evaluator = AgentEvaluator()

    config = Config()

    data = load_yaml(kwargs["test_set_path"])
    
    evaluator.ragas_eval(
        data,
        InferenceEngineBuilder.build_inference_engine(
            config.get("inference_engine")).as_langchain_llm()
    )