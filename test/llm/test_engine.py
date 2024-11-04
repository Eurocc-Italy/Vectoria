from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.llm.inference_engine.huggingface_inference_engine import HuggingFaceInferenceEngine
from threading import Thread



def test_huggingface_inference_engine():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")

    engine_1 = HuggingFaceInferenceEngine(config.get("inference_engine"))
    engine_2 = HuggingFaceInferenceEngine(config.get("inference_engine"))

    assert engine_1 == engine_2

    llm_1 = engine_1.as_langchain_llm()
    llm_2 = engine_2.as_langchain_llm()

    assert llm_1 == llm_2

    response_1 = llm_1("Hello, world!")
    response_2 = llm_2("Hello, world!")

    assert response_1 == response_2


