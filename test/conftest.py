import os
import requests
import pytest
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

@pytest.fixture(scope="function")
def clear_inference_engine_cache():
    InferenceEngineBuilder.clear_cache()

def pytest_addoption(parser):
    parser.addoption("--davinci", action="store_true", help="Use davinci configuration")
    parser.addoption("--leonardo", action="store_true", help="Use leonardo configuration")

@pytest.fixture(scope="session", autouse=True)
def set_configuration_file(request):
    if request.config.getoption("--davinci"):
        os.environ["VECTORIA_CONFIG_FILE"] = str(TEST_DIR / "data" / "config" / "davinci_hpc.yaml")
    elif request.config.getoption("--leonardo"):
        os.environ["VECTORIA_CONFIG_FILE"] = str(TEST_DIR / "data" / "config" / "leonardo_hpc.yaml")
    else:
        raise ValueError("Please specify the configuration file to use with --davinci or --leonardo")

@pytest.fixture(scope="session")
def data_dir():
    return TEST_DIR / "data"

@pytest.fixture(scope="session")
def docx_test_file(data_dir):
    return data_dir / "docx" / "test_docx.docx"

@pytest.fixture(scope="session")
def index_test_folder(data_dir):
    return data_dir / "index" / "BAAI__bge-m3_faiss_index"

@pytest.fixture(scope="function")
def config():
    return Config().load_config(os.environ["VECTORIA_CONFIG_FILE"])

@pytest.fixture(scope="session", autouse=True)
def vllm_server_status_fn():
    return _ping_vllm_server

def _ping_vllm_server(engine_config: dict):
    response = requests.get(engine_config.get("url").replace("/v1", "/version"))
    return response.status_code == 200

@pytest.fixture(scope="session", autouse=True)
def ollama_server_status_fn():
    return _ping_ollama_server

def _ping_ollama_server(engine_config: dict):
    # TODO: Add ollama server status check
    #response = requests.get(engine_config.get("url").replace("/v1", "/version"))
    #return response.status_code == 200
    return False