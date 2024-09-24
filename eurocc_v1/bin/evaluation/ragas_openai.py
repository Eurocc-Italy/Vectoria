import os 
os.environ["OPENAI_API_KEY"] = "sk-proj-GWmRyOB4lCfpnotiF5wsT3BlbkFJf9FvlkHdBA4CU6qv2KSC"
from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

from ragas import evaluate


def load_dataset():
    # Example data samples
    data_samples = {
        'question': [
            'When was the first super bowl?', 
            'Who won the most super bowls?'
        ],
        'answer': [
            'The first superbowl was held on Jan 15, 1967', 
            'The most super bowls have been won by The New England Patriots'],
        'contexts': [
            [
                'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
            ],
            [
                'The Green Bay Packers...Green Bay, Wisconsin.', 
                'The Packers compete...Football Conference'
            ]
        ],
        'ground_truth': [
            'The first superbowl was held on January 15, 1967', 
            'The New England Patriots have won the Super Bowl a record six times'
        ]
    }

    dataset = Dataset.from_dict(data_samples)

    return dataset


result = evaluate(
    load_dataset(),
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)
df = result.to_pandas()
df.head()

