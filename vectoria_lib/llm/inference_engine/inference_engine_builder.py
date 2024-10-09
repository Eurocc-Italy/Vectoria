from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase
from vectoria_lib.llm.inference_engine.huggingface_inference_engine import HuggingFaceInferenceEngine
from vectoria_lib.llm.inference_engine.ollama_inference_engine import OllamaInferenceEngine
from vectoria_lib.llm.inference_engine.openai_inference_engine import OpenAIInferenceEngine

class InferenceEngineBuilder:

    @staticmethod
    def build_inference_engine(args: dict) -> InferenceEngineBase:
        if args["name"] == "huggingface":
            return HuggingFaceInferenceEngine(args)
        elif args["name"] == "ollama":
            return OllamaInferenceEngine(args)
        elif args["name"] == "openai":
            return OpenAIInferenceEngine(args)
        else:
            raise ValueError(f"Unknown inference engine: {args['name']}")