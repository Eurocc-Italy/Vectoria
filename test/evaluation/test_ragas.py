import pytest
from vectoria_lib.evaluation.tools.ragas_eval import RagasEval
from vectoria_lib.common.config import Config
from vectoria_lib.common.paths import TEST_DIR
from vectoria_lib.common.io.file_io import get_file_io
from vectoria_lib.llm.inference_engine.inference_engine_builder import InferenceEngineBuilder
from ragas.run_config import RunConfig
from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    NonLLMStringSimilarity,
    RougeScore,
    FactualCorrectness,
    SemanticSimilarity
)

@pytest.mark.xfail(reason="Hugging Face evaluation raise ../aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [0,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && index out of bounds` failed.")
def test_ragas_hugging_face():
    pass

@pytest.mark.slow
@pytest.mark.parametrize(
    "metric", 
    [   
        NonLLMStringSimilarity(),
        RougeScore(rogue_type="rougeL", measure_type="fmeasure"),
        LLMContextRecall(),
        LLMContextPrecisionWithoutReference()
    ]
)
@pytest.mark.skipif(True, reason="TODO: ping vllm server")
def test_ragas_vllm(metric):

    ragas_eval = RagasEval(
        dict(
            metrics = [metric]
        )
    )

    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    
    scores = ragas_eval.eval(
        
        get_file_io("yaml").read(TEST_DIR / "data" / "eval" / "qa.yaml"),
        
        InferenceEngineBuilder.build_inference_engine(
            dict(
                name='openai',
                model_name='hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                url="http://localhost:8899/v1",
                api_key="abcd"
            ),
        ).as_langchain_llm(),
        
        run_config = RunConfig(
            timeout = 60
        )
    )

    assert metric.name in scores



@pytest.mark.xfail(reason="Bug of trailing slashes")
def test_ragas_factual_correctness(metric):

    ragas_eval = RagasEval(
        dict(
            metrics = [FactualCorrectness(mode="precision", atomicity="low", coverage="low")]
        )
    )

    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    
    scores = ragas_eval.eval(
        
        get_file_io("yaml").read(TEST_DIR / "data" / "eval" / "qa.yaml"),
        
        InferenceEngineBuilder.build_inference_engine(
            dict(
                name='openai',
                model_name='hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                url="http://localhost:8899/v1",
                api_key="abcd"
            ),
        ).as_langchain_llm(),
        
        run_config = RunConfig(
            timeout = 60
        )
    )

    assert metric.name in scores

@pytest.mark.skipif(True, reason="TODO: ping both vllm servers")
def test_ragas_semantic_similarity():

    ragas_eval = RagasEval(
        dict(
            metrics = [SemanticSimilarity()]
        )
    )

    config = Config()
    config.load_config(TEST_DIR / "data" / "config" / "test_config.yaml")
    
    scores = ragas_eval.eval(
        
        get_file_io("yaml").read(TEST_DIR / "data" / "eval" / "qa.yaml"),
        
        InferenceEngineBuilder.build_inference_engine(
            dict(
                name='openai',
                model_name='hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4',
                url="http://localhost:8899/v1",
                api_key="abcd"
            ),
        ).as_langchain_llm(),
        

        InferenceEngineBuilder.build_inference_engine(
            dict(
                name='openai',
                model_name='BAAI/bge-multilingual-gemma2',
                url="http://localhost:8898/v1",
                api_key="abcd"
            ),
        ).as_langchain_llm(),


        run_config = RunConfig(
            timeout = 60
        )
    )

    assert metric.name in scores