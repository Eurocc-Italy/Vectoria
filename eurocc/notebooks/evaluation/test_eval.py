import os
import sys
import pprint
import pandas as pd
import numpy as np
import ragas
import datasets

# Libraries to customize ragas critic model.
from ragas.llms import LangchainLLMWrapper
from langchain_community.chat_models import ChatOllama

# Libraries to customize ragas embedding model.
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

# Import the evaluation metrics.
from ragas.metrics import (
    context_recall, 
    context_precision, 
    faithfulness, 
    answer_relevancy, 
    answer_similarity,
    answer_correctness
)
from datasets import Dataset

def main():
    data_samples = {
        'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
        'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
        'contexts': [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'],
                     ['The Green Bay Packers...Green Bay, Wisconsin.', 'The Packers compete...Football Conference']],
        'ground_truth': ['The first superbowl was held on January 15, 1967', 'The New England Patriots have won the Super Bowl a record six times']
    }

    dataset = Dataset.from_dict(data_samples)
    
    ##########################################
    # Set the evaluation type.
    EVALUATE_WHAT = 'ANSWERS' 
    #EVALUATE_WHAT = 'CONTEXTS'
    ##########################################

    # Set the metrics to evaluate.
    if EVALUATE_WHAT == 'ANSWERS':
        eval_metrics = [
            answer_relevancy,
            answer_similarity,
            answer_correctness,
            faithfulness,
        ]
        metrics = ['answer_relevancy', 'answer_similarity', 'answer_correctness', 'faithfulness']
    elif EVALUATE_WHAT == 'CONTEXTS':
        eval_metrics = [
            context_recall, 
            context_precision, 
        ]
        metrics = ['context_recall', 'context_precision']
    
    # Change the default llm-as-critic model to gpt-3.5-turbo.
    # LLM_NAME = "gpt-3.5-turbo" #OpenAI
    # ragas_llm = ragas.llms.llm_factory(model=LLM_NAME)

    # Change the default the llm-as-critic model to local llama3.
    model_name = "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"#"meta-llama/Meta-Llama-3.1-8B-Instruct" #microsoft/Phi-3-mini-4k-instruct"#"swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)


    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        temperature=0.1, 
        repetition_penalty=1.1,
        max_new_tokens=2000
    )


    langchain_llm = HuggingFacePipeline(pipeline=pipe)


    #embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Wrap the LLM and embeddings with the RAGAS wrappers
    ragas_llm = LangchainLLMWrapper(langchain_llm)

    # Change the default embeddings models to use model on HuggingFace.
    EMB_NAME = "BAAI/bge-m3"
    #model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    lc_embed_model = HuggingFaceEmbeddings(
        model_name=EMB_NAME,
        #model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    ragas_emb = LangchainEmbeddingsWrapper(embeddings=lc_embed_model)

    # Change embeddings and critic models for each metric.
    for metric in metrics:
        globals()[metric].llm = ragas_llm
        globals()[metric].embeddings = ragas_emb

    # Define function to evaluate RAGAS model.
    def evaluate_ragas_model(data_samples, ragas_eval_metrics, what_to_evaluate='CONTEXTS'):
        """Evaluate the RAGAS model using the input dataset."""
        
        if what_to_evaluate == 'ANSWERS':
            eval_metrics = [
                answer_relevancy,
                #answer_similarity,
                #answer_correctness,
                #faithfulness,
            ]
        elif what_to_evaluate == 'CONTEXTS':
            eval_metrics = [
                context_recall, 
                context_precision,
            ]

        ragas_results = ragas.evaluate(data_samples, metrics=eval_metrics,
                                        #is_async=False,
                                        llm = ragas_llm,
                                        embeddings =ragas_emb )
        print(ragas_results)
        return ragas_results

    # Execute the evaluation.
    print(f"Evaluating {EVALUATE_WHAT} using eval questions:")
    ragas_result = evaluate_ragas_model(
        dataset, 
        eval_metrics, 
        what_to_evaluate=EVALUATE_WHAT
    )

if __name__ == "__main__":
    main()
