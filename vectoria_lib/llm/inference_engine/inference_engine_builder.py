#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from vectoria_lib.llm.inference_engine.huggingface_inference_engine import HuggingFaceInferenceEngine
from vectoria_lib.llm.inference_engine.openai_inference_engine import OpenAIInferenceEngine
from vectoria_lib.common.utils import Singleton

class InferenceEngineBuilder(Singleton):
    """
    Builds an inference engine objects and caches them.
    """
    CACHE = {}
    logger = logging.getLogger("llm")

    @staticmethod
    def build_inference_engine(
        args: dict  
    ) -> InferenceEngineBase:
        name = args["name"]
        model_name = args["model_name"]

        if f"{name}-{model_name}" in InferenceEngineBuilder.CACHE:
            InferenceEngineBuilder.logger.info("Returning cached inference engine for %s-%s", name, model_name)
            return InferenceEngineBuilder.CACHE[f"{name}-{model_name}"].update_args(args)
        
        if name == "huggingface":
            inference_engine = HuggingFaceInferenceEngine(args)
        elif name == "openai":
            inference_engine = OpenAIInferenceEngine(args)
        else:
            raise ValueError(f"Unknown inference engine: {name}")
        
        InferenceEngineBuilder.CACHE[f"{name}-{model_name}"] = inference_engine

        InferenceEngineBuilder.logger.info(
            "Caching inference engine for %s-%s. Cache size: %d. Cache keys: %s", name, model_name, len(InferenceEngineBuilder.CACHE), InferenceEngineBuilder.CACHE.keys())

        return inference_engine

    @staticmethod
    def clear_cache():
        InferenceEngineBuilder.CACHE = {}
        InferenceEngineBuilder.logger.info("Cleared inference engine cache")