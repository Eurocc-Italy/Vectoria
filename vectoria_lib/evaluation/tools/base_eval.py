from abc import ABC, abstractmethod
import logging

class BaseEval(ABC):

    def __init__(self):
        self.logger = logging.getLogger('evaluation')

    @abstractmethod
    def _convert_data_format_for_eval_tool(self, data: dict) -> dict:
        pass

    @abstractmethod
    def eval(self, data: dict) -> list[dict]:
        pass