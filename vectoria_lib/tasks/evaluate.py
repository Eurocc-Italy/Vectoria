#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
# ----------------------------------------------------------------------------------------------
from pathlib import Path
import time
import logging
from vectoria_lib.llm.agent_evaluator import AgentEvaluator
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.config import Config

def load_json(path: str):
    import json
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data

def evaluate(
    **kwargs: dict
):

    evaluator = AgentEvaluator(
        output_root_path = Path(kwargs["test_set_path"]).parent,
        test_set_name    = Path(kwargs["test_set_path"]).stem
    )

    config = Config()
    logger = logging.getLogger('db_management')

    start_time = time.perf_counter()
    data = load_json(kwargs["test_set_path"])
    logger.debug("Loading test_set %s took %.2f seconds", kwargs["test_set_path"], time.perf_counter() - start_time)
    
    start_time = time.perf_counter()

    evaluator.ragas_eval(
        data,
        InferenceEngineBuilder.build_inference_engine(
            config.get("inference_engine")
        ).as_langchain_llm()
        # InferenceEngineBuilder.build_inference_engine(
        #     dict(
        #         name='huggingface',
        #         model_name='BAAI/bge-m3',
        #         device="cuda",
        #     )
        # ).as_langchain_llm()
    )
    logger.debug("Ragas evaluation took %.2f seconds", time.perf_counter() - start_time)

# if __name__ == "__main__":
#     Config().load_config("/home/leobaro/workspace/labs/vectoria-project/led_use_case/config_docx.yaml")
#     evaluate(
#         test_set_path = "/home/leobaro/workspace/labs/vectoria-project/led_use_case/data/annotated/dataset_v1_annotated.json"
#     )