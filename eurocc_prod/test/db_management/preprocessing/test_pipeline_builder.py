import pytest

from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder

def test_pipeline():
    config = Config() 
    config.set("chunk_size", 100)
    config.set("chunk_overlap", 10)
    config.set("documents_format", "docx")

    pipeline = PreprocessingPipelineBuilder.build_pipeline(config)

    assert pipeline is not None
    assert pipeline.text_cleaner is not None
    assert pipeline.text_extractor is not None
    assert pipeline.chunking is not None