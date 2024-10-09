from dataclasses import dataclass, field

import logging
logger = logging.getLogger('db_management')

@dataclass
class DocumentData:
    """
    A dataclass to store the extracted data from a document.

    unstructured_text (str): The unstructured text extracted from the document.
    structured_text (list): The structured text extracted from the document. 
                            This is a list of dictionaries with the following keys:
                                - name: the name of the paragraph. [TODO: should go in metadata]
                                - docs: a list of langchain.docstore.document.Document objects.
                                - childs: a list of dictionaries with the same structure. 
    """
    unstructured_text: str = ""
    structured_text: list = field(default_factory=list)

def get_structured_data(doc_data: DocumentData):
    logger.debug("Getting structured data from doc")
    return doc_data.structured_text