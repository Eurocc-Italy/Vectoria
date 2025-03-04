import logging
from abc import ABC, abstractmethod
from pathlib import Path
from langchain.docstore.document import Document

class VectorStoreBase(ABC):

    def __init__(self):
        self.logger = logging.getLogger("rag")

    @abstractmethod
    def make_index(self, docs: list[Document]):
        pass

    @abstractmethod
    def dump_to_disk(self, output_path: str | Path = ".", output_suffix: str = ""):
        pass

    @abstractmethod
    def load_from_disk(self, input_path: str | Path):
        pass

    @abstractmethod
    def as_retriever(self, **kwargs):
        pass

    @abstractmethod
    def search(self, query: str, **kwargs):
        pass

    @abstractmethod
    def is_empty(self):
        pass
