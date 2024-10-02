import argparse
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR

def test_default_config():
    config = Config()
    assert config.get("retriever_top_k") == 5
    assert config.get("inference_engine")["model_name"] == "meta-llama/Meta-Llama-3.1-8B-Instruct"

def test_custom_config():
    config = Config().load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    assert config.get("retriever_top_k") == 5
    assert config.get("inference_engine")["device"] == "cuda"

def test_override_config():
    config = Config()
    config.update_from_args(argparse.Namespace(retriever_top_k=10))
    assert config.get("retriever_top_k") == 10
    config.set("log_level", "INFO")
    assert config.get("log_level") == "INFO"
