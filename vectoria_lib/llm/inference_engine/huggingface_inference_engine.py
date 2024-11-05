from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from vectoria_lib.llm.inference_engine.custom_hf.huggingface_reranker import HuggingFaceReranker

from langchain_core.language_models.llms import BaseLanguageModel

from langchain_huggingface import ChatHuggingFace


import time
import logging 

class HuggingFaceInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Hugging Face: 

    HuggingFace:
    TextGenerationPipeline init parameters: https://huggingface.co/docs/transformers/v4.45.1/en/main_classes/pipelines#transformers.TextGenerationPipeline
    TextGenerationPipeline call parameters: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/pipelines/text_generation.py#L215
        https://huggingface.co/docs/transformers/en/generation_strategies

    How to
    https://python.langchain.com/docs/how_to/#chat-models

    
    TODO: HuggingFaceInferenceEngine does not inherit from InferenceEngineBase anymore because of the metaclass Singleton.
    """
    def __init__(self, args: dict):
        super().__init__(args)
        
        self.logger = logging.getLogger('llm')

        start_time = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args["model_name"],
            trust_remote_code = self.args["trust_remote_code"]
        )
        self.logger.debug("Loading tokenizer took %.2f seconds", time.perf_counter() - start_time)

        quantization_config = None
        if self.args["load_in_8bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
        elif self.args["load_in_4bit"]:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True
            )
        else:
            self.args["device"] = None

        start_time = time.perf_counter()
        
        # TODO: device_map="auto" will not deallocate memory between tests
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args["model_name"],
            quantization_config = quantization_config,
            device_map = self.args["device_map"],
            trust_remote_code = self.args["trust_remote_code"]
            #device = self.args["device"]
        )
        self.pipe = None
        self.logger.debug("Loading model %s took %.2f seconds", self.args["model_name"], time.perf_counter() - start_time)
        self.model.eval()


    def as_langchain_llm(self) -> BaseLanguageModel:
        # https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace
        # return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=self.pipe))
        self.pipe = pipeline(
            "text-generation", 
            model = self.model,
            tokenizer = self.tokenizer,
            max_new_tokens = self.args["max_new_tokens"], # https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
            return_full_text = False,
            temperature = self.args["temperature"]
        )
        return HuggingFacePipeline(pipeline=self.pipe)
    
    def as_langchain_reranker_llm(self) -> BaseLanguageModel:
        return HuggingFaceReranker(model=self.model, tokenizer=self.tokenizer)