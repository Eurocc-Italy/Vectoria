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
        start_keywords = ["AI:", "RISPOSTA:", "risposta:"]
        for keyword in start_keywords:
            if keyword in text:
                logger.debug("Filter prefix match '%s'!", keyword)
                return text.split(keyword)[1].strip()
        logger.debug("Filter prefix no match")
        return text
        
    def filter_postfix(self, text: str):
        stop_keywords = ["FINE", "END"]
        for keyword in stop_keywords:
            if keyword in text:
                logger.debug("Filter postfix match '%s'!", keyword)
                parts = text.split(keyword)
                # monkey-patch for EN qa.txt prompt. The 'END' token is placed
                # both at the beginning and at the end of the answer. 
                sorted_parts = sorted(parts, key=len, reverse=True)
                return sorted_parts[0].strip()
        logger.debug("Filter postfix no match")
        return text

    def parse(self, text: str) -> str:
        logger.debug("CustomResponseParser: parsing:\n %s", text)
        return self.filter_postfix(self.filter_postfix(text))
