# https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/chains 

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents.base import _validate_prompt

from typing import Any, Dict, Union

from langchain_core.retrievers import (
    BaseRetriever,
)
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("{page_content}")

def format_docs(inputs: dict) -> str:
    """
    We assume that the retriver returns a dictionary with the "docs" key.
    DEFAULT_DOCUMENT_PROMPT is the prompt used for formatting each document into a string. 
    Input variables can be "page_content" or any metadata keys that are in all documents. 
    "page_content" will automatically retrieve the `Document.page_content`, and all other 
    inputs variables will be automatically retrieved from the `Document.metadata` dictionary. 
    Default to a prompt that only contains `Document.page_content`.    
    """
    return "\n\n".join(
        format_document(doc, DEFAULT_DOCUMENT_PROMPT)
        for doc in inputs["docs"]
    )


def create_qa_chain(
    prompt: BasePromptTemplate,
    llm: LanguageModelLike,
    output_parser: BaseOutputParser,
    *,
    retriever: Optional[BaseRetriever] = None,
    reranker: Optional[Any] = None
) -> Runnable[Dict[str, Any], Any]:
    """
    Create a chain for passing a list of Documents to a model.
    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context" (override by
            setting document_variable), which will be used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.
    """
    _validate_prompt(prompt, "context") # Check if prompt has 'context' key

    if retriever:
        retrieval_chain = ( (lambda x: x["input"]) | retriever ).with_config(run_name="retrieval_chain")

    combine_docs_chain = RunnableLambda(format_docs).with_config(run_name="combine_docs_chain")

    generation_chain = (prompt | llm | output_parser).with_config(run_name="generation_chain")

    # How it works:
    # Eventually we'll call the invoke() method passing the "input" key.
    # The RunnablePassthrough allows us to pass the input "key" not only to the next runnable
    # but also as key of the final output dictionary.
    # We then assign the retrieval_chain to the "docs" key and the combine_docs_chain to the "context" key.
    # Then we pass the input dictionary to the prompt. Still, these key will be available in the final output dictionary.
    # Finally, we assign the (prompt | llm | output_parser) to the "answer" key. We're not interested in the output 
    # of the prompt and llm.
    chain = RunnablePassthrough()
    
    if retriever:
        chain = chain.assign(docs=retrieval_chain)
    
    chain = chain.assign(context=combine_docs_chain)
    
    chain = chain.assign(answer=generation_chain)
    
    return chain

