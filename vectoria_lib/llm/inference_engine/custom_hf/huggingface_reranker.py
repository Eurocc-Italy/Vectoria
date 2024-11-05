from __future__ import annotations  # type: ignore[import-not-found]


import importlib.util
import torch
import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.messages import SystemMessage
from transformers import PreTrainedTokenizerBase, PreTrainedModel
DEFAULT_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_BATCH_SIZE = 4

logger = logging.getLogger("llm")

class HuggingFaceReranker(BaseLLM):

    model: PreTrainedModel = None  #: :meta private:
    tokenizer: PreTrainedTokenizerBase = None  #: :meta private:

    model_id: Optional[str] = None
    model_kwargs: Optional[dict] = None
    tokenizer_kwargs: Optional[dict] = None
    batch_size: int = DEFAULT_BATCH_SIZE

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_id": self.model_id,
            "model_kwargs": self.model_kwargs,
            "tokenizer_kwargs": self.tokenizer_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "huggingface_reranker"

    def argsort(self, seq):
        # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
        return sorted(range(len(seq)), key=seq.__getitem__, reverse=True)
    
    def _convert_base_messages_to_pairs(self, base_messages: List[SystemMessage]) -> List[List[str]]:
        base_messages_str = base_messages.pop(0)
        base_messages = [message for message in base_messages_str.split("System: ") if message]
        return [ [base_message for base_message in base_messages[i:i+2]] for i in range(0, len(base_messages), 2)]
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:

        yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        scores: List[float] = []
        for i in range(0, len(prompts), self.batch_size):

            batch_prompts = prompts[i : i + self.batch_size]
            
            pairs = self._convert_base_messages_to_pairs(batch_prompts)    

            # pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]

            # Process batch of prompts
            # responses = self.pipeline(
            #     batch_prompts,
            #     **pipeline_kwargs,
            # )


            with torch.no_grad():
                inputs = self._get_inputs(pairs)
                scores.extend(self.model(**inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float())
        
        scores = self.argsort([s.item() for s in scores])

        return LLMResult(
            generations=[[{"text": str(scores)}]] # :( 
        )
    
    def _get_inputs(self, pairs, prompt=None, max_length=1024):
        if prompt is None:
            prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        sep = "\n"
        prompt_inputs = self.tokenizer(prompt,
                                return_tensors=None,
                                add_special_tokens=False)['input_ids']
        sep_inputs = self.tokenizer(sep,
                            return_tensors=None,
                            add_special_tokens=False)['input_ids']
        inputs = []
        for query, passage in pairs: # [[q,a1], [q,a2]]
            query_inputs = self.tokenizer(f'A: {query}',
                                    return_tensors=None,
                                    add_special_tokens=False,
                                    max_length=max_length * 3 // 4,
                                    truncation=True)
            passage_inputs = self.tokenizer(f'B: {passage}',
                                    return_tensors=None,
                                    add_special_tokens=False,
                                    max_length=max_length,
                                    truncation=True)
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs['input_ids'],
                sep_inputs + passage_inputs['input_ids'],
                truncation='only_second',
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )
            item['input_ids'] = item['input_ids'] + sep_inputs + prompt_inputs
            item['attention_mask'] = [1] * len(item['input_ids'])
            inputs.append(item)

        return self.tokenizer.pad(
                inputs,
                padding=True,
                max_length=max_length + len(sep_inputs) + len(prompt_inputs),
                pad_to_multiple_of=8,
                return_tensors='pt',
        )
