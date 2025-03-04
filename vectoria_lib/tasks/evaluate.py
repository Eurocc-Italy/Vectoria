#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from pathlib import Path
import time
import logging
from vectoria.vectoria_lib.llm.evaluation.qa_application_evaluator import QAApplicationEvaluator
from vectoria_lib.llm.llm_builder import LLMFactory
from vectoria_lib.common.config import Config

def load_json(path: str):
    import json
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data

def evaluate(
    **kwargs: dict
):

    evaluator = QAApplicationEvaluator(
        output_root_path = Path(kwargs["test_set_path"]).parent,
        test_set_name    = Path(kwargs["test_set_path"]).stem,
        evaluation_tool  = kwargs["evaluation_tool"]
    )

    config = Config()
    logger = logging.getLogger('rag')

    start_time = time.perf_counter()
    data = load_json(kwargs["test_set_path"])
    logger.debug("Loading test_set %s took %.2f seconds", kwargs["test_set_path"], time.perf_counter() - start_time)
    
    start_time = time.perf_counter()

    evaluator.eval(
        data,
        LLMFactory.build_llm(
            config.get("inference_engine")
        ).as_langchain_llm()
        # LLMFactory.build_llm(
        #     dict(
        #         name='huggingface',
        #         model_name='BAAI/bge-m3',
        #         device="cuda",
        #     )
        # ).as_langchain_llm()
    )
    logger.debug("Evaluation took %.2f seconds", time.perf_counter() - start_time)
