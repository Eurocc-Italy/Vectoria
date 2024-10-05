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
        
    def filter_postfix(self, text: str):
        stop_keywords = ["Fine Risposta", "Fine", "FINE", "Human:"]
        for keyword in stop_keywords:
            if keyword in text:
                return text.split(keyword)[0].strip()
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

