# https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/chains 

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents.base import _validate_prompt

import logging
from typing import Any, Dict, Union

from langchain_core.retrievers import (
    BaseRetriever,
)
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

from vectoria_lib.db_management.reranking.reranker_base import BaseReranker
from vectoria_lib.llm.inference_engine.inference_engine_base import InferenceEngineBase

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")

logger = logging.getLogger("llm")

def format_docs(inputs: dict) -> str:
    """
    We assume that the retriver returns a dictionary with the "docs" key.
    DEFAULT_DOCUMENT_PROMPT is the prompt used for formatting each document into a string. 
    Input variables can be "page_content" or any metadata keys that are in all documents. 
    "page_content" will automatically retrieve the `Document.page_content`, and all other 
    inputs variables will be automatically retrieved from the `Document.metadata` dictionary. 
    Default to a prompt that only contains `Document.page_content`.    
    """
    if "reranked_docs" in inputs:
        docs = inputs["reranked_docs"]
    else:
        docs = inputs["docs"]

    formatted_docs = "\n\n".join(
        f"{i+1}. {format_document(doc, DEFAULT_DOCUMENT_PROMPT)}"
        for i, doc in enumerate(docs)
    )
    logger.debug("format_docs: formatted_docs: %s", formatted_docs)
    return formatted_docs

def reindex_docs(inputs: dict) -> dict:
    inputs["reranked_docs"] = [inputs["docs"][i] for i in inputs["reranked_docs_indices"]]
    return inputs

def create_qa_chain(
    prompt: BasePromptTemplate,
    llm: InferenceEngineBase,
    output_parser: BaseOutputParser,
    *,
    retriever_config: Optional[Dict[str, Any]] = None,
    reranker_config: Optional[Dict[str, Any]] = None
    
) -> Runnable[Dict[str, Any], Any]:
    """
    Create a chain for passing a list of Documents to a model.
    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context" (override by
            setting document_variable), which will be used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
        retriever_config: Configuration for the retriever.
        reranker_config: Configuration for the reranker.
    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.
    """
    _validate_prompt(prompt, "context") # Check if prompt has 'context' key

    if retriever_config:
        retrieval_chain = ( (lambda x: x["input"]) | retriever_config["retriever"].as_langchain_retriever() ).with_config(run_name="retrieval_chain")

    if reranker_config:
        reranking_chain = RunnablePassthrough()
        reranking_docs_indexes_chain = (reranker_config["prompt"] | reranker_config["inference_engine"].as_langchain_llm() | reranker_config["output_parser"]).with_config(run_name="reranking_docs_indexes_chain")

        reranking_chain = reranking_chain.assign(docs_for_reranking=RunnableLambda(format_docs))
        reranking_chain = reranking_chain.assign(reranked_docs_indices=reranking_docs_indexes_chain)
        reranking_chain = reranking_chain.assign(reranked_docs=RunnableLambda(reindex_docs))
        reranking_chain = reranking_chain.with_config(run_name="reranking_chain")

    combine_docs_chain = RunnableLambda(format_docs).with_config(run_name="combine_docs_chain")

    generation_chain = (prompt | llm.as_langchain_llm() | output_parser).with_config(run_name="generation_chain")

    # How it works:
    # Eventually we'll call the invoke() method passing the "input" key.
    # The RunnablePassthrough allows us to pass the input "key" not only to the next runnable
    # but also as key of the final output dictionary.
    # We then assign the retrieval_chain to the "docs" key and the combine_docs_chain to the "context" key.
    # Then we pass the input dictionary to the prompt. Still, these key will be available in the final output dictionary.
    # Finally, we assign the (prompt | llm | output_parser) to the "answer" key. We're not interested in the output 
    # of the prompt and llm.
    chain = RunnablePassthrough()
    
    if retriever_config:
        chain = chain.assign(docs=retrieval_chain)

    if reranker_config:
        chain = chain.assign(reranked_docs=reranking_chain)

    chain = chain.assign(context=combine_docs_chain)
    
    chain = chain.assign(answer=generation_chain)
    
    return chain

