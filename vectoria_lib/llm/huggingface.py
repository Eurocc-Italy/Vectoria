#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time, logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from vectoria_lib.llm.llm_base import LLMBase
from typing import Optional

class HuggingFaceLLM(LLMBase):
    """
    Wrapper on HuggingFace.

    TextGenerationPipeline init parameters: https://huggingface.co/docs/transformers/v4.45.1/en/main_classes/pipelines#transformers.TextGenerationPipeline
    TextGenerationPipeline call parameters: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/pipelines/text_generation.py#L215
        https://huggingface.co/docs/transformers/en/generation_strategies

    How to
    https://python.langchain.com/docs/how_to/#chat-models
    """

    def __init__(self, args: dict):
        """
        Initialize HuggingFace model with detailed configuration and logging.
        
        Args:
            args (Dict[str, Any]): Configuration for model loading
        """
        super().__init__(args)
        self.pipe = None
        self._initialize_model()


    def _initialize_model(self):
        """
        Centralized method for model and tokenizer initialization with performance tracking.
        """
        start_time = time.perf_counter()
        
        self._load_tokenizer()
        
        quantization_config = self._get_quantization_config()
        if quantization_config is None:
            self.args["device"] = None
        
        start_time = time.perf_counter()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args["model_name"],
                quantization_config=quantization_config,
                device_map=self.args.get("device_map", "auto"),
                trust_remote_code=self.args.get("trust_remote_code", False),
            )
            self.model.eval()
            self.logger.debug(
                "Model %s loading time: %.2f seconds", self.args["model_name"], time.perf_counter() - start_time
            )
        except Exception as e:
            self.logger.error("Failed to load model: %s", e)
            raise


    def _load_tokenizer(self):
        """
        Load the tokenizer for the model.
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args["model_name"], 
                trust_remote_code=self.args.get("trust_remote_code", False)
            )
        except Exception as e:
            self.logger.error("Failed to load tokenizer: %s", e)
            raise


    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """
        Determine appropriate quantization configuration based on model arguments.
        
        Returns:
            Optional[BitsAndBytesConfig]: Quantization configuration or None
        """
        if self.args.get("load_in_8bit"):
            return BitsAndBytesConfig(
                load_in_8bit=True, 
                bnb_8bit_compute_dtype=torch.bfloat16
            )
        elif self.args.get("load_in_4bit"):
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        return None


    def as_langchain_completion_model(self) -> BaseLanguageModel:
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.args["max_new_tokens"],
            return_full_text=False,
            temperature=self.args["temperature"]
        )
        return HuggingFacePipeline(pipeline=self.pipe)

    def as_langchain_chat_model(self) -> BaseLanguageModel:
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.args["max_new_tokens"],
            return_full_text=False,
            temperature=self.args["temperature"]
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=self.pipe))

    def as_langchain_embeddings_model(self) -> Embeddings:
        return HuggingFaceEmbeddings(model_name=self.args["model_name"])
