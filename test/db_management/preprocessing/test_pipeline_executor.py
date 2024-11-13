import os
import pytest
from time import time
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR


from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_executor import PreprocessingPipelineExecutor
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder

@pytest.mark.parametrize("multiproc", [False, True])
def test_pipeline_executor(config, multiproc):
    config.set("pp_multiprocessing", multiproc)
    config.config["pp_steps"][6]["chunk_size"] = 12
    config.config["pp_steps"][6]["chunk_overlap"] = 4

    pipeline = PreprocessingPipelineBuilder.build_pipeline()
    
    t = time()
    processed_docs = pipeline.run(TEST_DIR / "data/docx")
    print("Time taken:", time() - t)

    # A list of LangChain Document (chunks) for each input document
    assert len(processed_docs) == 73
    