import json, yaml, time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
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
from langchain_core.language_models.llms import BaseLanguageModel
from vectoria_lib.llm.agents.qa import QAAgent

class AgentEvaluator:

    def __init__(self, output_root_path: str | Path, test_set_name: str):
        self.answers = []
        self.retrived_context = []
        self.questions = None
        self.ground_truth = None
        self.output_root_path = output_root_path
        self.test_set_name = test_set_name
        self.logger = logging.getLogger('llm')

    def generate_answers(
            self,
            agent: QAAgent,
            test_set_path: str | Path,
        ) -> Path:
        
        self.logger.info("Generating answers for %s", test_set_path)

        with open(test_set_path, 'r', encoding='utf-8') as file:
            test_set = yaml.safe_load(file)

        self.questions = test_set["question"]
        self.ground_truth = test_set["ground_truth"]

        for q in tqdm(self.questions):
            q,a,c = agent.ask(q)
            self.answers.append(a)
            self.retrived_context.append(c)

        data = {
            'user_input': self.questions, 
            'response': self.answers,
            'retrieved_contexts': [[d.page_content for d in docs] for docs in self.retrived_context], 
            'reference': self.ground_truth
        }

        return self.dump(
            data,
            output_root_path=self.output_root_path,
            name="answers_and_contexts",
            out_format="json",
            add_time_stamp=True
        )
    
    

    def dump(
            self, 
            data: dict,
            output_root_path: str | Path,
            name: str,
            out_format: str = "json", 
            add_time_stamp: bool = True
        ) -> Path:
        output_path = Path(output_root_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if add_time_stamp:
            name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"

        if out_format == "json":
            name = f"{name}.json"
            with open(output_path / name, 'w', encoding='utf-8') as file:
                json.dump(data, file)
        elif out_format == "yaml":
            name = f"{name}.yaml"
            with open(output_path / name, 'w', encoding='utf-8') as file:
                yaml.dump(data, file)
        elif out_format == "txt":
            name = f"{name}.txt"
            with open(output_path / name, 'w', encoding='utf-8') as file:
                file.write(str(data))
        else:
            raise ValueError(f"Invalid format: {out_format}")
        
        return output_path
    
    def ragas_eval(
            self,
            eval_data: dict,
            evaluator_llm: BaseLanguageModel,
            embeddings_llm: BaseLanguageModel = None
        ) -> dict:
        eval_dataset = self._to_ragas_format(eval_data)
        evaluator_llm = LangchainLLMWrapper(evaluator_llm)
        metrics = [
            # Context Retrieval
            LLMContextRecall(),
            LLMContextPrecisionWithoutReference(),

            # Natural Language Comparison
            #FactualCorrectness(mode="precision", atomicity="low", coverage="low") # TODO: bug of 
            #SemanticSimilarity()
            
            NonLLMStringSimilarity(),
            #BleuScore(weights=(0.25, 0.25, 0.25, 0.25)), # TODO: AssertionError: The number of hypotheses and their reference(s) should be the same
            RougeScore(rogue_type="rougeL", measure_type="fmeasure")
        ]

        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=embeddings_llm,
            run_config = RunConfig(
                timeout=360
            )
        )
        
        scores = results.scores.to_pandas() # TODO: results is now a dict without mean and std

        scores.dropna(inplace=True)
        
        # TODO: make mean() and std() for each metric
    
        self.logger.info(f"Ragas evaluation results: {scores}")

        self.dump(
            scores.to_dict(),
            output_root_path=self.output_root_path,
            name=f"{self.test_set_name}_ragas_eval_results",
            add_time_stamp=False,
            out_format="yaml"
        )
        
        metrics_means, metrics_stds = self._get_mean_std(scores.to_dict())
        print(f"Mean: {metrics_means}, Std: {metrics_stds}")

        output_path, output_name = self.output_root_path, f"{self.test_set_name}_ragas_eval_results_bar_plot.png"
        self._make_bar_plot(metrics_means, metrics_stds, output_path, output_name)

        return results


    def _to_ragas_format(self, eval_data: dict) -> EvaluationDataset:
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



    def _get_mean_std(self, data: dict) -> tuple[float, float]:
        # Returns a tuple (mean, std) for each metric
        metrics_means = {}
        metrics_stds = {}
        for metric_name, metric_values in data.items():
            metrics_means[metric_name] = np.mean(list(metric_values.values()))
            metrics_stds[metric_name] = np.std(list(metric_values.values()))
        return metrics_means, metrics_stds

    def _make_bar_plot(self, metrics_means: dict, metrics_stds: dict, output_path: Path, output_name: str):
        plt.rcParams.update({
            'axes.labelsize': 14,     # Axis labels
            'axes.titlesize': 16,     # Title
            'xtick.labelsize': 12,    # X-axis tick labels
            'ytick.labelsize': 12,    # Y-axis tick labels
            'font.size': 12           # General text font size
        })        
        metrics = list(metrics_means.keys())
        mean_values = list(metrics_means.values())
        std_values = list(metrics_stds.values())

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar plot with error bars
        bars = ax.bar(x, mean_values, width, yerr=std_values, capsize=5, label='Mean')

        # Add labels, title, and custom x-axis tick labels
        ax.set_ylabel('Scores')
        ax.set_title('Metrics Mean with Standard Deviation')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        # Display the plot
        plt.tight_layout()
        plt.show()

        fig.savefig(output_path / output_name)
