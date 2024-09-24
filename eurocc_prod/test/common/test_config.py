import argparse
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR

def test_default_config():
    config = Config()
    assert config.get("retriever_top_k") == 5
    assert config.get("llm_model") == "meta-llama/Meta-Llama-3.1-8B-Instruct"
    assert config.get("max_new_tokens") == 150

def test_custom_config():
    config = Config().load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    assert config.get("retriever_top_k") == 5
    assert config.get("embedder_device") == "cpu"
    assert config.get("device") == "cpu"

def test_override_config():
    config = Config()
    config.update_from_args(argparse.Namespace(retriever_top_k=10))
    assert config.get("retriever_top_k") == 10

