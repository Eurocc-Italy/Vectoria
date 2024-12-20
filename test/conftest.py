import os, sys
venv_path=os.environ["VIRTUAL_ENV"]
python_version=sys.version[0:4]
lib_path=venv_path + "/lib/python" + python_version + "/site-packages"

# strip it from sys.path
sys.path.remove(lib_path)

# restore it in the right position
sys.path.insert(1,lib_path)

import requests
import pytest
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config

@pytest.fixture(scope="function")
def clear_inference_engine_cache():
    InferenceEngineBuilder.clear_cache()

def pytest_addoption(parser):
    pass

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
def config(request):
    c = Config().load_config(TEST_DIR / "data" / "config" / "test.yaml")
    c.set("vectoria_logs_dir", value=str(TEST_DIR / "logs" / request.node.name))
    return c

@pytest.fixture(scope="session", autouse=True)
def vllm_server_status_fn():
    return _ping_vllm_server

def _ping_vllm_server(engine_config: dict):
    try:
        response = requests.get(engine_config.get("url").replace("/v1", "/version"))
        return response.status_code == 200
    except Exception as e:
        return False

@pytest.fixture(scope="session", autouse=True)
def ollama_server_status_fn():
    return _ping_ollama_server

def _ping_ollama_server(engine_config: dict):
    # TODO: Add ollama server status check
    #response = requests.get(engine_config.get("url").replace("/v1", "/version"))
    #return response.status_code == 200
    return False