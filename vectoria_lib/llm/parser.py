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
        start_keywords = ["RISPOSTA:", "ANSWER:"]
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
                for p in parts:
                    if p.strip():
                        return p.strip()
        logger.debug("Filter postfix no match")
        return text

    def parse(self, text: str) -> str:
        #logger.debug("CustomResponseParser: parsing:%s", text)
        return self.filter_postfix(self.filter_prefix(text))

