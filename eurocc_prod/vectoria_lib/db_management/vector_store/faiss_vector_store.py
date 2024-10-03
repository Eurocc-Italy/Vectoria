#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from pathlib import Path
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from vectoria_lib.common.config import Config

class FaissVectorStore:
    """
    A wrapper around the FAISS library to create and manage a FAISS-based vector store.

    This class provides methods to create a FAISS index from documents, retrieve the index as a retriever, 
    and serialize/deserialize the index using pickle.
    """

    # Reference to FAISS integration documentation:
    # https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/
    # https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.faiss.FAISS.html

    @staticmethod
    def load_from_pickle(pkl_path: str | Path):
        """
        Load a FAISS index from a serialized pickle file.

        Parameters:
        - pkl_path (str | Path): The path to the serialized FAISS index pickle file.
        
        Returns:
        - FaissVectorStore: An instance of FaissVectorStore with the loaded FAISS index.

        """
        pkl_path = Path(pkl_path)
        pkl_bytes = pkl_path.read_bytes()

        # BAAI__bge-m3_faiss_index.pkl -> BAAI/bge-m3
        model_name = pkl_path.stem.split("_faiss_index")[0].replace('__', '/')

        fvs = FaissVectorStore(model_name)
        index = FAISS.deserialize_from_bytes(
            embeddings=fvs.hf_embedder,
            serialized=pkl_bytes,
            allow_dangerous_deserialization=True
        )
        fvs.index = index
        
        return fvs

    def __init__(self, model_name: str = None):
        """
        Initialize a FaissVectorStore object.

        Parameters:
        - model_name (str): The name of the embedding model to use.
        """
        config = Config()

        self.model_name = model_name

        if model_name is None:
            self.model_name = config.get("hf_embedder_model_name")

        # FIXME: solo per test vado a modificare il path del modello da caricare per usare quello locale
        # lascio "self.model_name" invariato per non alterare il nome del dump
        # local_model = "/leonardo_work/PhDLR_prod/bge-m3" # CINECA

        self.hf_embedder = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            #model_name=local_model,
            model_kwargs={
                "device": config.get("embedder_device")
            },
            encode_kwargs={
                "normalize_embeddings": config.get("normalize_embeddings")
            }
        )
        self.index = None

    def make_index(self, docs: list[str]):
        """
        Create a FAISS index from a list of documents.

        Parameters:
        - docs (list[str]): A list of documents to be indexed.

        Returns:
        - FaissVectorStore: The FaissVectorStore instance with the created index.
        """
        self.index = FAISS.from_documents(docs, self.hf_embedder)
        return self

    def as_retriever(self, **kwargs):
        """
        Convert the FAISS index into a retriever object.

        Parameters:
        - kwargs: Additional keyword arguments to configure the retriever.

        Returns:
        - Retriever: A retriever object based on the FAISS index.
        """

        if self.index is None:
            raise ValueError("Index is not created. Call make_index() first.")
            
        return self.index.as_retriever(**kwargs)

    def dump_to_pickle(self, output_path: str | Path = ".") -> Path:
        """
        Serialize the FAISS index to a pickle file.

        Parameters:
        - output_path (str | Path): The directory where the pickle file should be saved. Default is the current directory.

        Returns:
        - Path: The path to the saved pickle file.
        """
        pkl = self.index.serialize_to_bytes()
        model_name = self.model_name.replace('/','__') # repo/name => repo__name
        pkl_path = Path(output_path) / f"{model_name}_faiss_index.pkl"
        with open(pkl_path, "wb") as f:
            f.write(pkl)
        return pkl_path

        
