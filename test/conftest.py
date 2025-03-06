import os, sys
try:
  venv_path=os.environ["VIRTUAL_ENV"]
  python_version=sys.version[0:4]
  lib_path=venv_path + "/lib/python" + python_version + "/site-packages"
  # strip it from sys.path
  sys.path.remove(lib_path)
  # restore it in the right position
  sys.path.insert(1,lib_path)
except Exception:
  pass

import requests
import pytest
from vectoria_lib.llm.llm_factory import LLMFactory
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.config import Config
from vectoria_lib.common.io.file_io import get_file_io

@pytest.fixture(scope="function")
def clear_inference_engine_cache():
    LLMFactory.clear_cache()

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
    return data_dir / "index" / "attention_is_all_you_need_index"

@pytest.fixture()
def eval_data_qa():
    return get_file_io("json").read(TEST_DIR / "data" / "eval" / "qa.json")

@pytest.fixture(scope="function")
def config(request, write_config_file_for_tests):
    c = Config().load_config(write_config_file_for_tests)
    c.set("vectoria_logs_dir", value=str(TEST_DIR / "logs" / request.node.name))
    return c

@pytest.fixture(scope="session", autouse=True)
def openai_server_status_fn():
    return _ping_openai_server

def _ping_openai_server(engine_config: dict):
    try:
        response = requests.get(engine_config.get("openai_api_base").replace("/v1", "/version"))
        return response.status_code == 200
    except Exception:
        return False
    

@pytest.fixture(scope="session")
def write_config_file_for_tests(request):
    config_file_for_tests_path = "/tmp/test.yaml"
    with open(config_file_for_tests_path, "w") as f:
        f.write("""
vectoria_logs_dir: /tmp/vectoria_logs
log_level: DEBUG
langchain_tracking: false
system_prompts_lang: eng

langfuse:
  enabled: true
  host: http://localhost:3000
  public_key: pk-lf-c3ba5270-9449-4ba0-b3c7-fcd0a2393785
  secret_key: sk-lf-80663bf5-c73c-4921-81b0-fe3c8b5648a3
                
data_ingestion:
  multiprocessing: true
  
  extraction:
    format: docx
    dump_doc_structure_on_file: true
    regexes_for_metadata_extraction:
      - name: DOC_ID
        pattern: '^Document Title'

  regexes_for_replacement:
    - name: remove_multiple_spaces
      pattern: '[ \t]{2,}'
      replace_with: ' '
    - name: remove_bullets
      pattern: '^\s*[\u2022\u25AA\u27A2]\s*'
      replace_with: ''
    - name: remove_ligature_st
      pattern: 'ﬆ'
      replace_with: 'st'


  chunking:
    chunk_size: 512
    chunk_overlap: 256
    separators: ["\n\n", "\n", " ", ""]
    is_separator_regex: [false, false, false, false]
    dump_chunks_on_file: true

vector_store:
  name: faiss
  model_name: BAAI/bge-m3
  device: cuda
  normalize_embeddings: false

retriever:
  enabled: false
  name: faiss
  search_type: 'mmr'
  k: 5
  fetch_k: 5
  lambda_mult: 0.5

reranker:
  enabled: false
  rerank_k: 3
  inference_engine:
    name: huggingface
    url: null
    api_key: null
    model_name: BAAI/bge-reranker-base
    device: cuda
    load_in_4bit: false
    load_in_8bit: false
    max_new_tokens: 150
    trust_remote_code: false
    device_map: null
    temperature: 0.1

full_paragraphs_retriever:
  enabled: false

inference_engine:
  name: huggingface
  url: null
  api_key: null
  model_name: HuggingFaceTB/SmolLM-135M
  device: cuda
  load_in_4bit: false
  load_in_8bit: false
  max_new_tokens: 10
  trust_remote_code: false
  device_map: null
  do_sample: false
  temperature: 0

""")
    return config_file_for_tests_path