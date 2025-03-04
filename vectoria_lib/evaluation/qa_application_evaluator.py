#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import yaml, logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

from vectoria_lib.applications.qa import QAApplication
from vectoria.vectoria_lib.common.io.file_io import get_file_io
from .tools.ragas_eval import ragas_evaluation
from vectoria_lib.common.io.file_io import get_file_io
from vectoria_lib.common.plots import make_bar_plot

class QAApplicationEvaluator:

    def __init__(self, output_root_path: str | Path, test_set_name: str):
        self.answers = []
        self.retrived_context = []
        self.questions = None
        self.ground_truth = None
        self.output_root_path = output_root_path
        self.test_set_name = test_set_name
        self.logger = logging.getLogger('evaluation')

    def generate_answers(
            self,
            app: QAApplication,
            test_set_path: str | Path,
        ) -> Path:
        
        self.logger.info("Generating answers for %s", test_set_path)

        with open(test_set_path, 'r', encoding='utf-8') as file:
            test_set = yaml.safe_load(file)

        self.questions = test_set["question"]
        self.ground_truth = test_set["ground_truth"]

        for q in tqdm(self.questions):
            q,a,c = app.ask(q)
            self.answers.append(a)
            self.retrived_context.append(c)

        data = {
            'user_input': self.questions, 
            'response': self.answers,
            'retrieved_contexts': [[d.page_content for d in docs] for docs in self.retrived_context], 
            'reference': self.ground_truth
        }

        return get_file_io("json").write(
            data,
            output_root_path=self.output_root_path,
            name="answers_and_contexts",
            add_time_stamp=True
        )
    
    def evaluate(
            self,
            eval_data: dict,
            *args
        ) -> dict:

        scores: list[dict] = ragas_evaluation(
            eval_data,
            *args
        )

        metrics_means, metrics_stds = self._compute_mean_and_stddev(scores)

        output_file = get_file_io("yaml").write(
            scores,
            output_root_path=self.output_root_path,
            name=f"{self.test_set_name}_ragas_eval_results",
            add_time_stamp=False
        )
        output_path, output_name = self.output_root_path, f"{self.test_set_name}_ragas_eval_results_bar_plot.png"
        make_bar_plot(metrics_means, metrics_stds, output_path, output_name)

        return output_file



    def _compute_mean_and_stddev(self, metrics_dict):
        # {'context_recall': {0: 1.0, 1: 0.0, 2: 1.0}, 'llm_context_precision_without_reference': {0: 0.9999999999, 1: 0.9999999999, 2: 0.99999999995} }
        metrics_means = {}
        metrics_stds = {}
        for metric, values_dict in metrics_dict.items():
            values = list(values_dict.values())
            metrics_means[metric] = np.mean(values)
            metrics_stds[metric] = np.std(values)
        return metrics_means, metrics_stds
    