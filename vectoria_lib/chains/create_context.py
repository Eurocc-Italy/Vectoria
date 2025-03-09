from typing import List

from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate, format_document

def get_create_context_chain():
    """
    This chain is used to format the documents into a string.
    
    Required keys:
        * docs: list of langchain Document objects to be formatted
    """
    return (
                
        RunnableLambda(_format_docs)
        
    ).with_config(run_name="combine_docs_chain")


def _format_docs(inputs: dict) -> str:
    """
    This function formats the documents into a string.
    It relies on the langchain `format_document` function that can automatically
    retrieve the `Document.page_content` and the `Document.metadata` dictionary.
    """
    if "docs" not in inputs:
        return ""

    docs: List[Document] = inputs["docs"]

    CONCATENATION_SEPARATOR = "\n\n"
    DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template("""
"{page_content}
[Document name:'{doc_file_name}' Paragraph: '{paragraph_number}-{paragraph_name}']")
""")
    
    formatted_docs = CONCATENATION_SEPARATOR.join(
        f"{i+1}. {format_document(doc, DEFAULT_DOCUMENT_PROMPT)}"
        for i, doc in enumerate(docs)
    )
    return formatted_docs