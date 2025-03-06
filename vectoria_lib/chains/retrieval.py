from langchain_core.runnables import RunnableLambda

from vectoria_lib.common.config import Config
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory

def get_retrieval_chain():
    """
    This function creates a retrieval chain.
    It takes a retriever configuration and builds a retriever from it.
    It then returns a retrieval chain.
    """
    config = Config()
    
    retriever = VectorStoreFactory.create_vector_store(config.get("vector_store", "name")).as_retriever(
        search_config = config.get("retriever")
    )

    return (

        RunnableLambda(lambda x: x["input"]) |
        
        retriever
        
    ).with_config(run_name="retrieval_chain")

    




