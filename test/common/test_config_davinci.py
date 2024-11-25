import os
import pytest
import argparse
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR

def test_get_default_config(config):
    assert config.get("retriever", "top_k") == 5
    assert config.get("inference_engine", "model_name") == "meta-llama/Meta-Llama-3.1-8B-Instruct"

def test_get_custom_config(config):
    assert config.get("retriever", "top_k") == 5
    assert config.get("inference_engine", "device") == "cuda"

def test_get_not_valid_config(config):
    with pytest.raises(KeyError):
        config.get("top_k")
    with pytest.raises(KeyError):
        config.get("not_valid_key")

def test_set_config(config):
    config.set("retriever", "top_k", 10)
    assert config.get("retriever", "top_k") == 10

@pytest.mark.skip(reason="This feature is temporarily disabled")
def test_override_config(config):
    config.update_from_args(argparse.Namespace(top_k=10))
    assert config.get("retriever", "top_k") == 10
    config.set("log_level", "INFO")
    assert config.get("log_level") == "INFO"
