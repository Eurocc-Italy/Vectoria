import os
import pytest
from vectoria_lib.evaluation.tools.ragas_eval import ragas_evaluation
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.io.file_io import get_file_io
from vectoria_lib.llm.llm_factory import LLMFactory
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    NonLLMStringSimilarity,
    RougeScore,
    FactualCorrectness,
    SemanticSimilarity,
    Faithfulness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

@pytest.mark.xfail(reason="Hugging Face evaluation raise ../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && index out of bounds` failed.")
def test_ragas_hugging_face():
    pass

@pytest.mark.slow
@pytest.fixture
def metrics_with_llm(config: Config, vllm_server_status_fn):
    
    if not vllm_server_status_fn(config.get("evaluation", "inference_engine")):
        pytest.skip("VLLM server is not running")
    
    generation_llm = LangchainLLMWrapper(LLMFactory.build_llm(
            config.get("evaluation", "inference_engine")
        ).as_langchain_chat_model())
    
    metrics = [ 
        NonLLMStringSimilarity(),
        RougeScore(measure_type="fmeasure"),
        LLMContextRecall(llm=generation_llm),
        LLMContextPrecisionWithoutReference(llm=generation_llm),
        FactualCorrectness(llm=generation_llm, mode="precision", atomicity="low", coverage="low")
    ]
    return metrics

@pytest.mark.slow
@pytest.fixture
def metrics_with_embeddings(config: Config, vllm_server_status_fn):
    if not vllm_server_status_fn(config.get("evaluation", "embeddings_engine")):
        pytest.skip("VLLM server (embeddings_engine) is not running")
    
    embeddings_llm = LangchainEmbeddingsWrapper(LLMFactory.build_llm(
            config.get("evaluation", "embeddings_engine")
        ))
    
    metrics = [
        SemanticSimilarity(embeddings=embeddings_llm)
    ]
    return metrics

@pytest.fixture
def eval_data_qa():
    return get_file_io("yaml").read(TEST_DIR / "data" / "eval" / "qa.yaml")


def test_ragas_vllm_llm(metrics_with_llm, eval_data_qa):

    scores = ragas_evaluation(
        eval_data_qa,
        metrics=metrics_with_llm,
        run_config = RunConfig(
            timeout = 20
        )
    )

    for metric in metrics_with_llm:
        assert metric.name in scores
        assert bool(scores[metric.name])

@pytest.mark.skip(reason="Embeddings are not supported yet by vllm: 'VLLMInferenceEngine' object has no attribute 'aembed_documents'")
def test_ragas_vllm_embeddings(metrics_with_embeddings, eval_data_qa):

    scores = ragas_evaluation(
        eval_data_qa,
        metrics=metrics_with_embeddings,
        run_config = RunConfig(
            timeout = 20
        )
    )

    for metric in metrics_with_embeddings:
        assert metric.name in scores
        assert bool(scores[metric.name])
