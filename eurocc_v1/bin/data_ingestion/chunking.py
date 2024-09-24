from langchain.text_splitter import CharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size=200, size_overlap=0, separator=' '):
        """
        Initialize the TextChunker object by creating an instance of CharacterTextSplitter.

        Parameters:
        - chunk_size (int): The size of each chunk. Default is 200.
        - size_overlap (int): The overlap size between consecutive chunks. Default is 0 (no overlap).
        - separator (str): The character or substring used to split the text into chunks. Default is ' ' (space).
        """
        self.chunk_size = chunk_size
        self.size_overlap = size_overlap
        self.separator = separator
       
        self.text_splitter = CharacterTextSplitter(separator=self.separator, chunk_size=self.chunk_size, chunk_overlap=self.size_overlap)


    def chunk_text(self, text, mode='fixed_size'):
        """
        Chunk the input text into smaller segments based on the specified mode.

        Parameters:
        - text (str): The input text to be chunked.
        - mode (str): The chunking mode. Currently supported modes:
                      - 'fixed_size': Chunks the text into fixed-size segments based on chunk_size.
                                      Note that by default the last chunk contains the remaining characters 
                                      that couldn't fill a complete chunk of chunk_size. No padding. 
                      (Future modes will be added here)

        Returns:
        - list: A list of chunks where each chunk is a segment of the input text.
        """

        if mode == 'fixed_size':
            chunks = self.text_splitter.split_text(text)
        else:
            raise ValueError(f"Unsupported mode '{mode}'. Currently supported modes: 'fixed_size'.")

        return chunks
