import os
import pytest
import tempfile
import argparse
from pathlib import Path
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR

def test_get_default_config(config):
    assert config.get("retriever", "k") == 5
    assert config.get("inference_engine", "model_name") == "HuggingFaceTB/SmolLM-135M"

def test_load_custom_config(config):
    config_file_path = "/tmp/test_custom.yaml"
    with open(config_file_path, "w", encoding="utf-8") as f:
        f.write("log_level: ERROR\n")
        f.write("vectoria_logs_dir: /tmp/logs")
    config.load_config(config_file_path)
    assert config.get("log_level") == "ERROR"

def test_get_not_valid_config(config):
    with pytest.raises(KeyError):
        config.get("top_k")
    with pytest.raises(KeyError):
        config.get("not_valid_key")

def test_set_config(config):
    config.set("retriever", "k", 10)
    assert config.get("retriever", "k") == 10

@pytest.mark.skip(reason="This feature is temporarily disabled")
def test_override_config(config):
    config.update_from_args(argparse.Namespace(k=10))
    assert config.get("retriever", "k") == 10
    config.set("log_level", "INFO")
    assert config.get("log_level") == "INFO"
