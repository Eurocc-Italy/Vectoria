#
# VECTORIA
#
# @authors : Andrea Proia, Leonardo Baroncelli
#

# https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/chains 
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents.base import _validate_prompt
from langchain_core.messages import SystemMessage
import logging
from typing import Any, Dict
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from vectoria_lib.llm.llm_base import LLMBase
from langchain_core.documents import Document

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}\n[Document name:'{doc_file_name}' Paragraph: '{paragraph_number}-{paragraph_name}']")

logger = logging.getLogger("llm")

def get_correct_input_docs(inputs: dict) -> List[Document]:
    if "full_paragraphs_docs" in inputs:
        return inputs["full_paragraphs_docs"]
    elif "reranked_docs" in inputs:
        return inputs["reranked_docs"]
    else:
        return inputs["docs"]

def format_docs(docs: List[Document]) -> str:
    """
    We assume that the retriver returns a dictionary with the "docs" key.
    DEFAULT_DOCUMENT_PROMPT is the prompt used for formatting each document into a string. 
    Input variables can be "page_content" or any metadata keys that are in all documents. 
    "page_content" will automatically retrieve the `Document.page_content`, and all other 
    inputs variables will be automatically retrieved from the `Document.metadata` dictionary. 
    Default to a prompt that only contains `Document.page_content`.    
    """
    formatted_docs = "\n\n".join(
        f"{i+1}. {format_document(doc, DEFAULT_DOCUMENT_PROMPT)}"
        for i, doc in enumerate(docs)
    )
    return formatted_docs

def reindex_docs(inputs: dict, reranker_top_k: int) -> dict:
    llm_output = [inputs["docs"][i] for i in inputs["reranked_docs_indices"]]
    return llm_output[:reranker_top_k]

def create_reranking_input_pairs(inputs: dict) -> list[SystemMessage]:
    # The base_messages list will be converted into a list of strings:
    # [SystemMessage(content='what is panda?'), SystemMessage(content='hi'), SystemMessage(content='what is panda?'), BaseMessage(content='The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.')]
    # [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    base_messages = []
    for doc in inputs["docs"]:
        base_messages.append(SystemMessage(content=inputs["input"]))
        base_messages.append(SystemMessage(content=doc.page_content))

    return base_messages

def create_qa_chain(
    prompt: BasePromptTemplate,
    llm: LLMBase,
    output_parser: BaseOutputParser,
    *,
    retriever_config: Optional[Dict[str, Any]] = None,
    reranker_config: Optional[Dict[str, Any]] = None,
    full_paragraphs_retriever_config: Optional[Dict[str, Any]] = None
) -> Runnable[Dict[str, Any], Any]:
    """
    Create a chain for passing a list of Documents to a model.
    """
    _validate_prompt(prompt, "context") # Check if prompt has 'context' key

    if retriever_config:
        retrieval_chain = ( (lambda x: x["input"]) | retriever_config["retriever"].as_langchain_retriever() ).with_config(run_name="retrieval_chain")

    if reranker_config:
        reranking_chain = RunnablePassthrough()

        reranking_docs_indexes_chain = ( 
            RunnableLambda(create_reranking_input_pairs) | 
            reranker_config["reranker"].as_langchain_post_retrieval_step() | 
            RunnableLambda(lambda x: eval(x))
        ).with_config(run_name="reranking_docs_indexes_chain")

        reranking_chain = reranking_chain.assign(reranked_docs_indices=reranking_docs_indexes_chain)
        reranking_chain = reranking_chain.assign(reranked_docs=RunnableLambda(reindex_docs).bind(reranker_top_k=reranker_config["reranked_top_k"]))
        reranking_chain = reranking_chain.with_config(run_name="reranking_chain")

        reranking_chain = reranking_chain | RunnableLambda(lambda x: x["reranked_docs"])

    if full_paragraphs_retriever_config:
        full_paragraphs_retriever_chain = (
            RunnableLambda(get_correct_input_docs) | full_paragraphs_retriever_config["retriever"].as_langchain_post_retrieval_step()
        ).with_config(run_name="full_paragraphs_retriever_chain")

    combine_docs_chain = (RunnableLambda(get_correct_input_docs) | RunnableLambda(format_docs)).with_config(run_name="combine_docs_chain")

    generation_chain = (prompt | llm.as_langchain_completion_model() | output_parser).with_config(run_name="generation_chain")

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

    if full_paragraphs_retriever_config:
        chain = chain.assign(full_paragraphs_docs=full_paragraphs_retriever_chain)

    chain = chain.assign(context=combine_docs_chain)
    
    chain = chain.assign(answer=generation_chain)
    
    return chain

