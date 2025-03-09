from langchain_core.runnables import RunnableLambda
from vectoria_lib.components.vector_store.vector_store_factory import VectorStoreFactory
from vectoria_lib.components.postretrieval_steps.full_paragraphs import FullParagraphs
from vectoria_lib.common.config import Config

def get_context_enhancement_chain():

    config = Config()
    vector_store = VectorStoreFactory.create_vector_store(config.get("vector_store", "name"))
    full_paragraphs_retriever = FullParagraphs(vector_store)

    return (

        RunnableLambda(lambda x: x["docs"]) |

        full_paragraphs_retriever.as_langchain_post_retrieval_step()

    ).with_config(run_name="full_paragraphs_retriever_chain")
