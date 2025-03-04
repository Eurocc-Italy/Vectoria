import os
from pathlib import Path
import pytest
import langchain_core
from vectoria_lib.components.preprocessing.preprocessing_pipeline import PreprocessingPipeline

def test_build_preprocessing_pipeline(config):
    pipeline: PreprocessingPipeline = PreprocessingPipeline.build_pipeline()
    assert isinstance(pipeline.chain, langchain_core.runnables.base.RunnableSequence)
    assert pipeline.chain.first is not None
    assert len(pipeline.chain.middle) == 1
    assert pipeline.chain.last is not None

@pytest.mark.parametrize("multiproc", [False, True])
def test_run_preprocessing_pipeline(config, data_dir, multiproc):
    config.set("data_ingestion", "multiprocessing", multiproc)
    pipeline = PreprocessingPipeline.build_pipeline()    
    docs = [Path(data_dir / "docx" / doc_name) for doc_name in os.listdir(data_dir / "docx")]
    processed_docs = pipeline.run(docs)
    assert len(processed_docs) == 95
    