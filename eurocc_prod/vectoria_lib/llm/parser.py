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
        pattern = r"(?<=Risposta:)(.*)"
        return re.findall(pattern, text, re.DOTALL)[0]

        """
        match = re.search(r'Risposta:\s+(.*)', text, re.DOTALL)
        if match:
            logger.debug("Filter prefix match!")
            response = match.group(0).strip()
            return response
        logger.debug("Filter prefix no match")
        return None
        """
        
    def filter_postfix(self, text: str):
        match = re.search(r'(.+)(\s*Fine Risposta|Fine|Human:)', text)
        if match:
            logger.debug("Filter postfix match")
            response = match.group(1)
            response = re.sub(r'\s{2,}', ' ', response).strip()
            return response
        logger.debug("Filter postfix no match")
        return None

    def parse(self, text: str) -> str:
        response = self.filter_prefix(text)

        if not response:
            return f"[CustomResponseParser] failed to filter prefix for:\n {text}"

        response_without_suffix = self.filter_postfix(response)
        if not response_without_suffix:
            return response

        return response_without_suffix

