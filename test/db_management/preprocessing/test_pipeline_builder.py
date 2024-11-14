from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_builder import PreprocessingPipelineBuilder
from vectoria_lib.db_management.preprocessing.pipeline.preprocessing_pipeline_executor import PreprocessingPipelineExecutor

import langchain_core

def test_build_pipeline(config):
    pipeline: PreprocessingPipelineExecutor = PreprocessingPipelineBuilder.build_pipeline()
    assert isinstance(pipeline.chain, langchain_core.runnables.base.RunnableSequence)
    assert pipeline.chain.first is not None
    assert len(pipeline.chain.middle) == 0
    assert pipeline.chain.last is not None

