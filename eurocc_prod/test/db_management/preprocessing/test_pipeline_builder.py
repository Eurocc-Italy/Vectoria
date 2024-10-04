import pytest

from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline import PreprocessingPipeline
import langchain_core
def test_pipeline():
    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")

    pipeline: PreprocessingPipeline = PreprocessingPipelineBuilder.build_pipeline()
    
    assert isinstance(pipeline.chain, langchain_core.runnables.base.RunnableSequence)
    assert len(pipeline.chain.middle) == 5

