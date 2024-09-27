from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class HuggingFaceInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Hugging Face.
    """
    def __init__(self, args: dict):
        super().__init__(args)

        tokenizer = AutoTokenizer.from_pretrained(self.args["model_name"])
        model = AutoModelForCausalLM.from_pretrained(
            self.args["model_name"],
            load_in_8bit = self.args["load_in_8bit"]
        )
        if self.args["load_in_8bit"]:
            self.args["device"] = None
        
        self.pipe = pipeline(
            "text-generation", 
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = self.args["max_new_tokens"], # https://stackoverflow.com/questions/76772509/llama-2-7b-hf-repeats-context-of-question-directly-from-input-prompt-cuts-off-w
            device = self.args["device"]
        )
        
    def as_langchain_llm(self) -> HuggingFacePipeline:
        return HuggingFacePipeline(pipeline=self.pipe)
