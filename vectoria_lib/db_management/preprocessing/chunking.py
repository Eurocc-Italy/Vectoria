#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#
from pathlib import Path
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from vectoria_lib.common.config import Config
logger = logging.getLogger('db_management')
config = Config()

# TODO: superflous function
def recursive_character_text_splitter(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] = ["\n\n", "\n", " ", ""],
    is_separator_regex: list[bool] = [False, False, False, False],
    dump_chunks_on_file: bool = False
) -> list[Document]:
    """
    Split the input text into chunks using RecursiveCharacterTextSplitter.

    Parameters:
        - doc (Document): The input document    
    - config:
        - chunk_size (int): The size of the chunks.
        - chunk_overlap (int): The overlap between chunks.
        - separators (list[str]): A list of separators.  
        - is_separator_regex (list[bool]): A list of booleans indicating if the separators are regexes.
        - dump_chunks_on_file (bool): If True, the chunks are dumped on a file.

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
        chunk.page_content = chunk.page_content

    if dump_chunks_on_file:
        # delete each character in the filename that is not a letter 
        doc_metadata_str = ''.join(char for char in doc.metadata["doc_file_name"] if char.isalpha())
        output_name = f'doc_{doc_metadata_str.lower()}'
        _log_chunks_on_file(chunks, Path(config.get("vectoria_logs_dir") / "chunks" / f"{output_name}_chunk.txt")) 

    return chunks

def _log_chunks_on_file(documents, file_path: Path):
    Path(file_path.parent).mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            print(doc, file=f)