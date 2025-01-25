#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging
from abc import ABC, abstractmethod
from langchain_core.language_models.llms import BaseLanguageModel
from langchain_core.embeddings import Embeddings

class InferenceEngineBase(ABC):
    def __init__(self, args: dict):
        self.args = args
        self.name = args.pop("name")
        self.logger = logging.getLogger('llm')

    def update_args(self, new_args: dict):
        for key, value in new_args.items():
            if key in self.args and self.args[key] != value:
                self.logger.info("Updating %s with %s", key, value)
                self.args[key] = value
        return self

    @abstractmethod
    def as_langchain_completion_model(self) -> BaseLanguageModel:
        pass

    @abstractmethod
    def as_langchain_chat_model(self) -> BaseLanguageModel:
        pass

    @abstractmethod
    def as_langchain_embeddings_model(self) -> Embeddings:
        pass
    
    def __repr__(self):
        return f"InferenceEngineBase for {self.name}"