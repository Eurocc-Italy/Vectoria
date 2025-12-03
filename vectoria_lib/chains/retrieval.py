from langchain_core.runnables import RunnableLambda

from vectoria_lib.common.config import Config
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory
from vectoria_lib.components.llm.llm_factory import LLMFactory
from langchain.retrievers import ContextualCompressionRetriever

def get_retrieval_chain():

    """
    This function creates a retrieval chain.
    It takes a retriever configuration and builds a retriever from it.
    It then returns a retrieval chain.
    """
    config = Config()
    
    retriever = VectorStoreFactory.build_vector_store(**config.get("vector_store")).as_retriever(
        search_config = config.get("retriever")
    )

    if config.get("retriever", "enable_rerank"):
        
        cross_encoder_reranker = LLMFactory.build_llm(config.get("retriever", "inference_engine")).as_langchain_reranker_model()
        cross_encoder_reranker.top_n = config.get("retriever", "rerank_k")
        
        retriever = ContextualCompressionRetriever(
            base_compressor=cross_encoder_reranker, base_retriever=retriever
        )


    return (

        RunnableLambda(lambda x: x["input"]) |
        
        retriever
        
    ).with_config(run_name="retrieval_chain")

    




