import pytest
from time import time
from vectoria_lib.common.config import Config

from vectoria_lib.rag.preprocessing.pipeline.preprocessing_pipeline_executor import PreprocessingPipelineExecutor
from vectoria_lib.rag.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder

@pytest.mark.parametrize("multiproc", [False, True])
def test_pipeline_executor(config, data_dir, multiproc):
    config.set("pp_multiprocessing", value=multiproc)
    config.set("vectoria_logs_dir", value="/home/leobaro/workspace/labs/vectoria-project/vectoria/test_pipeline_executor_logs")
    pipeline = PreprocessingPipelineBuilder.build_pipeline()
    
    t = time()
    processed_docs = pipeline.run(data_dir / "docx")
    print("Time taken:", time() - t)

    # A list of LangChain Document (chunks) for each input document
    assert len(processed_docs) == 95
    