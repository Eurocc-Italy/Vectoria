import os
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder

def test_inference_engine_builder(config):
    InferenceEngineBuilder.clear_cache()
    
    config.set("inference_engine", "name", "huggingface")

    config.set("inference_engine", "model_name", "HuggingFaceTB/SmolLM-135M")
    engine_1 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))

    config.set("inference_engine", "model_name", "HuggingFaceTB/SmolLM-135M")
    engine_2 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))
    
    config.set("inference_engine", "model_name", "BAAI/bge-reranker-base")
    engine_3 = InferenceEngineBuilder.build_inference_engine(config.get("inference_engine"))

    assert engine_1 == engine_2
    assert engine_1 != engine_3
    assert engine_2 != engine_3

    assert InferenceEngineBuilder.CACHE[f"{engine_1.name}-{engine_1.args.get('model_name')}"] == engine_1
    assert InferenceEngineBuilder.CACHE[f"{engine_2.name}-{engine_2.args.get('model_name')}"] == engine_2
    assert InferenceEngineBuilder.CACHE[f"{engine_3.name}-{engine_3.args.get('model_name')}"] == engine_3

    assert len(InferenceEngineBuilder.CACHE) == 2