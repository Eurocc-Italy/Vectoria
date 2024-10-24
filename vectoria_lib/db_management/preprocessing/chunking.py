#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
from pathlib import Path
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
    is_separator_regex: list[bool] = [False, False, False, False],
    log_in_folder: Path = None
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
        - log_in_folder (Path): The folder where to log the chunks structure.

    Returns:
    - list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=separators,
        is_separator_regex=is_separator_regex
    )

    # since we are processing one document at a time, we can safely return the first element of the list
    chunks = text_splitter.create_documents([doc.page_content])

    for chunk in chunks:
        chunk.metadata = doc.metadata.copy()
        chunk.page_content = "(Nome paragrafo: " + doc.metadata["name"] + ") " + chunk.page_content # TODO: add the name of the document to the chunk. Are we sure?

    # delete each character in the filename that is not a letter 
    doc_metadata_str = ''.join(char for char in doc.metadata["name"] if char.isalpha())
    output_name = f'doc_{doc.metadata["source"]}_chunk_{doc_metadata_str.lower()}_{doc.metadata["level"]}_{doc.metadata["id"]}'
    
    if log_in_folder is not None:
        Path(log_in_folder).mkdir(parents=True, exist_ok=True)  
        _log_chunks_on_file(chunks, Path(log_in_folder) / f"{output_name}_chunk.txt") 

    return chunks

def _log_chunks_on_file(documents, file_path: Path):
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            print(doc, file=f)