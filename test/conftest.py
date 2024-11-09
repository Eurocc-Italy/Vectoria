import os
import pytest
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from vectoria_lib.common.paths import TEST_DIR

@pytest.fixture(scope="function")
def clear_inference_engine_cache():
    InferenceEngineBuilder.clear_cache()

@pytest.fixture(scope="session", autouse=True)
def set_configuration_file(request):
    if request.config.getoption("--davinci"):
        os.environ["VECTORIA_CONFIG_FILE"] = str(TEST_DIR / "data" / "config" / "davinci_hpc.yaml")
    elif request.config.getoption("--leonardo"):
        os.environ["VECTORIA_CONFIG_FILE"] = str(TEST_DIR / "data" / "config" / "leonardo_hpc.yaml")
    else:
        raise ValueError("Please specify the configuration file to use with --davinci or --leonardo")
    
def pytest_addoption(parser):
    parser.addoption("--davinci", action="store_true", help="Use davinci configuration")
    parser.addoption("--leonardo", action="store_true", help="Use leonardo configuration")
