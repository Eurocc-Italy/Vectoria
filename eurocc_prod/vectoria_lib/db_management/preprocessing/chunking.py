#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunking:
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 256):
        """
        Initialize the Chunking object with a chunk size and overlap for the text splitter.

        Parameters:
        - chunk_size (int): The maximum size of each text chunk. Default is 1024.
        - chunk_overlap (int): The overlap size between consecutive chunks to ensure text continuity. Default is 256.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger('db_management')

    def make_chunks(self, text: str) -> list[str]:
        """
        Split the input text into chunks using RecursiveCharacterTextSplitter.

        The text is split into chunks in a way that aims to preserve semantic units like paragraphs and sentences.
        This method splits based on a list of separators such as paragraphs, lines, and spaces, trying to create 
        chunks that are as meaningful as possible within the specified chunk size.

        Parameters:
        - text (str): The input text to be chunked.

        Returns:
        - list[str]: A list of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        return text_splitter.create_documents([text])
        