from vectoria_lib.db_management.reranking.reranker_base import BaseReranker
from vectoria_lib.db_management.reranking.cos_sim_reranker import CosSimReranker
# from vectoria_lib.db_management.reranking.llm_reranker import LLMReranker
# from vectoria_lib.llm.inference_engine_builder import InferenceEngineBuilder

class RerankerBuilder:
    # TODO: DEPRECATED

    @staticmethod
    def build_reranker(args: dict) -> BaseReranker:
        if args is None:
            return None
        
        if args["name"] == "cos_sim":
            return CosSimReranker(args)
        # elif args["name"] == "llm":
        #     if "inference_engine" not in args:
        #         raise ValueError(f"Reranker is {args['name']}-based but the 'inference engine' key is not found in the configuration")            
        #     return LLMReranker(args, InferenceEngineBuilder.build_inference_engine(args["inference_engine"]))
        else:
            raise ValueError(f"Unknown Reranker: {args['name']}")
