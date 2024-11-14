import logging
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class PostRetrievalStepBase(ABC):
    def __init__(self):
        self.logger = logging.getLogger("db_management")

    @abstractmethod
    def post_process(self, chunks: List[Document]) -> List[Document]:
        pass

    def as_langchain_post_retrieval_step(self):
        return self.post_process