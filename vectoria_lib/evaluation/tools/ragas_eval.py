import logging

from ragas.metrics import (
    LLMContextRecall, 
    LLMContextPrecisionWithoutReference, 
    #Faithfulness, 
    #FactualCorrectness,
    #SemanticSimilarity,
    NonLLMStringSimilarity,
    #BleuScore,
    RougeScore
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from ragas.dataset_schema import EvaluationResult

def ragas_evaluation(
        eval_data: dict,
        run_config: RunConfig = None,
        metrics: list = None,
    ) -> list[dict]:
    
    logger = logging.getLogger('evaluation')

    eval_dataset = convert_data_format_for_eval_tool(eval_data)
    
    scores: EvaluationResult = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        run_config = run_config if run_config else RunConfig(
            timeout=360
        )
    )

    scores = scores.to_pandas().dropna().to_dict()

    scores.pop("user_input")
    scores.pop("retrieved_contexts")
    scores.pop("response")
    scores.pop("reference")
    # {'context_recall': {0: 1.0, 1: 0.0, 2: 1.0}, 'llm_context_precision_without_reference': {0: 0.9999999999, 1: 0.9999999999, 2: 0.99999999995} }

    logger.info("Evaluation results: %s", scores)

    return scores

def convert_data_format_for_eval_tool(eval_data: dict) -> EvaluationDataset:
    samples = []
    for i in range(len(eval_data['question'])):
        sample = SingleTurnSample(
            user_input = eval_data['question'][i],
            reference  = eval_data['ground_truth'][i],
            response   = eval_data['answer'][i],
            retrieved_contexts = eval_data['contexts'][i]
    )
        samples.append(sample)

    return EvaluationDataset(samples=samples)