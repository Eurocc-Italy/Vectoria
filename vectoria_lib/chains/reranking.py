from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda

from vectoria_lib.components.llm.llm_factory import LLMFactory
from vectoria_lib.components.postretrieval_steps.huggingface_reranker import Reranker
from vectoria_lib.common.config import Config

def get_reranking_chain():

    config = Config()

    reranker_llm = Reranker(LLMFactory.build_llm(config.get("reranker", "inference_engine")))

    reranking_chain = RunnablePassthrough()

    reranking_docs_indexes_chain = (

        RunnableLambda(_create_reranking_input_pairs)   |

        reranker_llm.as_langchain_post_retrieval_step() |

        RunnableLambda(lambda x: eval(x))

    ).with_config(run_name="reranking_docs_indexes_chain")

    reranking_chain = reranking_chain.assign(reranked_docs_indices=reranking_docs_indexes_chain)
    reranking_chain = reranking_chain.assign(reranked_docs=RunnableLambda(_reindex_docs).bind(reranker_top_k=config.get("reranker", "rerank_k")))
    reranking_chain = reranking_chain.with_config(run_name="reranking_chain")
    reranking_chain = reranking_chain | RunnableLambda(lambda x: x["reranked_docs"])

    return reranking_chain

def _create_reranking_input_pairs(inputs: dict) -> list[SystemMessage]:
    """
    This function creates a list of SystemMessage objects from the input documents.
    It takes a dictionary as input, which contains a list of documents.
    It then creates a list of SystemMessage objects, where each SystemMessage object contains a question and a document.
    The question is the input string, and the document is the page_content of the document.
    """
    base_messages = []
    for doc in inputs["docs"]:
        base_messages.append(SystemMessage(content=inputs["input"]))
        base_messages.append(SystemMessage(content=doc.page_content))
    return base_messages

def _reindex_docs(inputs: dict, reranker_top_k: int) -> dict:
    """
    This function reindexes the documents based on the reranked documents indices.
    It takes a dictionary as input, which contains a list of documents and the reranked documents indices.
    It then returns a list of documents, which are the reranked documents.
    """
    llm_output = [inputs["docs"][i] for i in inputs["reranked_docs_indices"]]
    return llm_output[:reranker_top_k]

