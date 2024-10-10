import json, yaml, time
import logging
from pathlib import Path

from tqdm import tqdm
import pandas as pd
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset, SingleTurnSample
from ragas.run_config import RunConfig
from vectoria_lib.llm.agents.qa import QAAgent

class AgentEvaluator:

    def __init__(self):
        self.answers = []
        self.retrived_context = []
        self.questions = None
        self.ground_truth = None
        self.output_root_path = None
        self.test_set_name = None
        self.logger = logging.getLogger('llm')
 
    def generate_answers(
            self, 
            agent: QAAgent, 
            test_set_path: str | Path, 
            dump: bool = True
        ) -> dict:
        
        self.logger.info(f"Generating answers for {test_set_path.name}")

        self.output_root_path, self.test_set_name = Path(test_set_path).parent, Path(test_set_path).stem

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

        if dump:
            return self.dump(data)
        
        return data
    

    def dump(self, data: dict) -> dict:
        output_path = self.output_root_path / f"{self.test_set_name}_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data, file)    
        
    def ragas_eval(self, evaluator_llm, eval_data: dict) -> pd.Dataframe:

        eval_dataset = self.to_ragas_format(eval_data)
        evaluator_llm = LangchainLLMWrapper(evaluator_llm)
        metrics = [LLMContextRecall()]

        results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=evaluator_llm,
            run_config = RunConfig(
                timeout=360
            )
        )

        df = results.to_pandas()
        print(df.head())
        return df

    def _to_ragas_format(self, eval_data: dict, limit: int = None) -> EvaluationDataset:
        if limit is None or limit < 1:
            limit = len(eval_data['user_input'])
        samples = []
        for i in range(len(eval_data['user_input']), limit):
            sample = SingleTurnSample(
                user_input = eval_data['user_input'][i],
                reference  = eval_data['reference'][i],
                response   = eval_data['response'][i],
                retrieved_contexts = eval_data['retrieved_contexts'][i]
        )
            samples.append(sample)

        return EvaluationDataset(samples=samples)

