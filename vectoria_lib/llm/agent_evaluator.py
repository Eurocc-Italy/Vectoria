import json, yaml
import time
import logging
from pathlib import Path

from tqdm import tqdm

class AgentEvaluator:

    def __init__(self, agent):
        self.agent = agent
        self.answers = []
        self.retrived_context = []
        self.questions = None
        self.ground_truth = None
        self.output_root_path = None
        self.test_set_name = None
        self.logger = logging.getLogger('llm')
 
    def evaluate(self, test_set_path: str | Path):

        self.output_root_path, self.test_set_name = Path(test_set_path).parent, Path(test_set_path).stem

        with open(test_set_path, 'r', encoding='utf-8') as file:
            test_set = yaml.safe_load(file)

        self.questions = test_set["question"]
        self.ground_truth = test_set["ground_truth"]

        for q in tqdm(self.questions):
            q,a,c = self.agent.ask(q)
            self.answers.append(a)
            self.retrived_context.append(c)
        
    def dump(self):
        output_path = self.output_root_path / f"{self.test_set_name}_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump({
                'question': self.questions, 
                'answer': self.answers,
                'context': [[d.page_content for d in docs] for docs in self.retrived_context], 
                'ground_truth': self.ground_truth
            }, file)
        
    def ragas_eval(self):
        pass