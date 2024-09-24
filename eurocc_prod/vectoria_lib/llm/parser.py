#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import re
import logging
from langchain.schema import BaseOutputParser

logger = logging.getLogger("llm")

class CustomResponseParser(BaseOutputParser):
    
    def filter_prefix(self, text: str):
        match = re.search(r'^(.+\s*=*\s*RISPOSTA\s\s*=*\s)(.+)', text, re.DOTALL)
        if match:
            logger.debug("Filter prefix match: %s", match.group(2))
            response = match.group(2)
            response = re.sub(r'\s{2,}', ' ', response).strip()
            return response
        return None
    
    def filter_postfix(self, text: str):
        match = re.search(r'(.+)(\s*Fine Risposta|Fine|Human:)', text)
        if match:
            logger.debug("Filter postfix match: %s", match.group(1))
            response = match.group(1)
            response = re.sub(r'\s{2,}', ' ', response).strip()
            return response
        return None

    def parse(self, text: str) -> str:
        response = self.filter_prefix(text)
        if not response:
            return f"No valid response found for text:\n {text}"
        response = self.filter_postfix(response)
        if not response:
            return f"\n {text}"
        return response
