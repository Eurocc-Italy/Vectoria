#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import json, yaml, time
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from langchain_core.language_models.llms import BaseLanguageModel

from vectoria_lib.llm.agents.qa import QAAgent
from vectoria.vectoria_lib.common.io.file_io import get_file_io
from vectoria_lib.common.plots import make_bar_plot
from tools.ragas_eval import RagasEval

class AgentEvaluator:

    def __init__(self, output_root_path: str | Path, test_set_name: str, evaluation_tool: str):
        self.answers = []
        self.retrived_context = []
        self.questions = None
        self.ground_truth = None
        self.output_root_path = output_root_path
        self.test_set_name = test_set_name
        self.logger = logging.getLogger('llm')
        if evaluation_tool == "ragas":
            self.eval_tool = RagasEval()

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

        return get_file_io("json").write(
            data,
            output_root_path=self.output_root_path,
            name="answers_and_contexts",
            add_time_stamp=True
        )
    
    
    def eval(
            self,
            eval_data: dict,
            *args
        ) -> dict:

        eval_data = self.eval_tool.convert_data_format_for_eval_tool(eval_data)
        return self.eval_tool.eval(
            eval_data,
            *args
        )