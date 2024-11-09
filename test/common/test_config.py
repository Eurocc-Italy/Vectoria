import os
import pytest
import argparse
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR

def test_get_default_config():
    config = Config().load_config(os.environ["VECTORIA_CONFIG_FILE"])
    assert config.get("retriever", "retriever_top_k") == 5
    assert config.get("inference_engine", "model_name") == "meta-llama/Meta-Llama-3.1-8B-Instruct"

def test_get_custom_config():
    config = Config().load_config(os.environ["VECTORIA_CONFIG_FILE"])
    assert config.get("retriever", "retriever_top_k") == 5
    assert config.get("inference_engine", "device") == "cuda"

def test_get_not_valid_config():
    config = Config()
    with pytest.raises(KeyError):
        config.get("retriever_top_k")
    with pytest.raises(KeyError):
        config.get("not_valid_key")

def test_set_config():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    config.set("retriever", "retriever_top_k", 10)
    assert config.get("retriever", "retriever_top_k") == 10

@pytest.mark.skip(reason="This feature is temporarily disabled")
def test_override_config():
    config = Config()
    config.load_config(os.environ["VECTORIA_CONFIG_FILE"])
    config.update_from_args(argparse.Namespace(retriever_top_k=10))
    assert config.get("retriever", "retriever_top_k") == 10
    config.set("log_level", "INFO")
    assert config.get("log_level") == "INFO"
