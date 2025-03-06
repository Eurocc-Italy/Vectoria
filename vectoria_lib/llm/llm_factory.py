#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from vectoria_lib.llm.llm_base import LLMBase
from vectoria_lib.llm.huggingface import HuggingFaceLLM
from vectoria_lib.llm.openai import OpenAILLM
from vectoria_lib.common.utils import Singleton
from typing import Dict, Type

class LLMFactory(Singleton):
    """
    A singleton cache manager for LLM instances
    """
    logger = logging.getLogger("llm")

    CACHE: Dict[str, LLMBase] = {}
    ENGINE_REGISTRY: Dict[str, Type[LLMBase]] = {
        "huggingface": HuggingFaceLLM,
        "openai": OpenAILLM
    }

    @classmethod
    def build_llm(cls, args: dict) -> LLMBase:

        cls.logger.info("Building LLM %s %s", args["name"], args["model_name"])
        name = args["name"]
        model_name = args["model_name"]
        cache_key = f"{name}-{model_name}"

        if cache_key in cls.CACHE:
            cls.logger.info("Returning cached inference engine for %s-%s", name, model_name)
            return cls.CACHE[cache_key].update_args(args)

        if name not in cls.ENGINE_REGISTRY:
            registered_engines = ", ".join(cls.ENGINE_REGISTRY.keys())
            raise ValueError(f"Unknown inference engine: {name}. Registered engines: {registered_engines}")
        
        inference_engine = cls.ENGINE_REGISTRY[name](args)
        cls.CACHE[cache_key] = inference_engine

        cls.logger.info(
            "Caching inference engine for %s. "
            "Cache size: %d. "
            "Cache keys: %s",
            cache_key,
            len(cls.CACHE),
            list(cls.CACHE.keys())
        )

        return inference_engine

    @classmethod
    def clear_cache(cls):
        """Clear the entire LLM cache and log the operation."""
        cls.CACHE.clear()
        cls.logger.info("Cleared inference engine cache")