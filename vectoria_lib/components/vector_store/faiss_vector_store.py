#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import pickle, logging
from pathlib import Path

from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from vectoria_lib.common.config import Config
from vectoria_lib.components.vector_store.vectore_store_base import VectorStoreBase

class FaissVectorStore(VectorStoreBase):
    """
    A wrapper around the FAISS library to create and manage a FAISS-based vector store.

    This class provides methods to create a FAISS index from documents, retrieve the index as a retriever, 
    and serialize/deserialize the index using pickle.
    
    Reference to FAISS integration documentation: https://python.langchain.com/docs/integrations/vectorstores/faiss
    API Reference: https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.faiss.FAISS.html

    """
    def __init__(
            self,
            model_name,
            device,
            normalize_embeddings,
            index_path = None
        ):
        super().__init__()

        self.model_name = model_name

        self.logger.info("Loading Embedder model: %s.." % self.model_name)

        self.hf_embedder = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": normalize_embeddings
            }
        )

        if index_path:
            self.index = FAISS.load_local(index_path, self.hf_embedder, allow_dangerous_deserialization=True)
        else:
            self.index = None


    def make_index(self, docs: list[Document]):
        """
        Create a FAISS index from a list of documents.

        Parameters:
        - docs (list[str]): A list of documents to be indexed.

        Returns:
        - FaissVectorStore: The FaissVectorStore instance with the created index.
        """
        self.logger.info("Creating FAISS index from %d documents.." % len(docs))
        self.index = FAISS.from_documents(docs, self.hf_embedder)
        return self

    def dump_to_disk(self, output_path: str | Path) -> Path:
        """
        Serialize the FAISS index.

        Parameters:
        - output_path (str | Path): The path to the directory where the FAISS index should be saved.
        Returns:
        - Path: The path to the saved FAISS index.
        """
        model_name = self.model_name.replace('/','__') # repo/name => repo__name
        Path(output_path).mkdir(exist_ok=True, parents=True)
        output_path = Path(output_path) / f"{model_name}_faiss_index"        
        self.logger.info("Serializing FAISS index to pickle file %s" % output_path)
        self.index.save_local(output_path)
        return output_path

    def load_from_disk(self, input_path: str | Path):
        if self.index:
            self.logger.info("Index already loaded. Skipping deserialization.")
            return self
        self.logger.info("Deserializing FAISS index from pickle file %s" % input_path)
        input_path = Path(input_path)
        self.index = FAISS.load_local(input_path, self.hf_embedder, allow_dangerous_deserialization=True)
        return self

    def as_retriever(self, search_config: dict):
        """
        Convert the FAISS index into a retriever object.

        Parameters:
        - kwargs: Additional keyword arguments to configure the retriever.

        Returns:
        - Retriever: A retriever object based on the FAISS index.
        """
        self.logger.info("Converting FAISS index to retriever with kwargs: %s" % search_config)
        if self.index is None:
            raise ValueError("Index is not created. Call 'make_index' or 'load_from_disk' first.")
            
        return self.index.as_retriever(
            search_type = search_config["search_type"],
            search_kwargs = {
                "k": search_config["k"],
                "fetch_k": search_config["fetch_k"],
                "lambda_mult": search_config["lambda_mult"]
            }
        )

    def search(self, query: str, **kwargs) -> list[Document]:
        """
        Parameters: https://python.langchain.com/docs/integrations/vectorstores/faiss/#query-directly
        """
        return self.index.similarity_search(
            query, **kwargs
        )
    
    def is_empty(self):
        return self.index is None

    def add_documents(self, docs: list[Document]):
        """
        Add new documents to the FAISS index.

        Parameters:
        - docs (list[Document]): A list of new documents to be indexed.
        """
        if self.index is None:
            raise ValueError("Index is not initialized. Call make_index() first.")

        self.logger.info("Adding %d new documents to FAISS index.." % len(docs))
        self.index.add_documents(docs)