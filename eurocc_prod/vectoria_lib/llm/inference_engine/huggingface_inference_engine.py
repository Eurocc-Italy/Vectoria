from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

class HuggingFaceInferenceEngine(InferenceEngineBase):
    """
    Wrapper on Hugging Face: 

    HuggingFace:
    TextGenerationPipeline init parameters: https://huggingface.co/docs/transformers/v4.45.1/en/main_classes/pipelines#transformers.TextGenerationPipeline
    TextGenerationPipeline call parameters: https://github.com/huggingface/transformers/blob/v4.45.1/src/transformers/pipelines/text_generation.py#L215
        https://huggingface.co/docs/transformers/en/generation_strategies

    How to
    https://python.langchain.com/docs/how_to/#chat-models

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
            repetition_penalty=1.03, # TODO: move in configuration
            device = self.args["device"],
            return_full_text = False
        )
        
    def as_langchain_llm(self) -> HuggingFacePipeline:
        # https://python.langchain.com/api_reference/huggingface/chat_models/langchain_huggingface.chat_models.huggingface.ChatHuggingFace.html#langchain_huggingface.chat_models.huggingface.ChatHuggingFace
        # ChatHuggingFace(llm=)
        return HuggingFacePipeline(pipeline=self.pipe)
