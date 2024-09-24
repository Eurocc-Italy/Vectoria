from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, encoder_name="BAAI/bge-m3"):
        """
        Initialize the TextEmbedder with a SentenceTransformer model for generating embeddings.

        Args:
        - encoder_name (str): Name of the SentenceTransformer model to use (default: "BAAI/bge-m3").
        """
        self.model = SentenceTransformer(encoder_name)

    def generate_embeddings(self, chunks):
        """
        Generate embeddings for a list of text chunks using the initialized SentenceTransformer model.

        Args:
        - chunks (list of str): List of text chunks for which embeddings are to be generated.

        Returns:
        - embeddings (list of numpy.ndarray): List of embeddings corresponding to input text chunks.
        """
        embeddings = self.model.encode(chunks)
        return embeddings
