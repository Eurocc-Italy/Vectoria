#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

logger = logging.getLogger('db_management')

# TODO: superflous function
def recursive_character_text_splitter(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] = ["\n\n", "\n", " ", ""],
    is_separator_regex: list[bool] = [False, False, False, False]
) -> list[Document]:
    """
    Split the input text into chunks using RecursiveCharacterTextSplitter.

    Parameters:
    - text (str): The input text.
    - config:
        - chunk_size (int): The size of the chunks.
        - chunk_overlap (int): The overlap between chunks.
        - separators (list[str]): A list of separators.  
        - is_separator_regex (list[bool]): A list of booleans indicating if the separators are regexes.

    Returns:
    - list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=is_separator_regex,
    )

    # since we are processing one document at a time, we can safely return the first element of the list
    return text_splitter.create_documents([doc.page_content])[0] 