#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import logging

class Cleaning:

    def __init__(self):
        """
        Initialize the Cleaning object.

        This sets up a logger for tracking cleaning operations and initializes an empty list to store
        the cleaning steps that will be applied to the text.
        """
        self.logger = logging.getLogger('db_management')
        self.cleaning_steps = []

    def add_cleaning_step(self, cleaning_step):
        """
        Add a cleaning step to the list of cleaning steps.

        Parameters:
        - cleaning_step (function): A function that defines a cleaning step, which will be applied
                                    to the input text in the clean_text method.
        
        Returns:
        - self: The Cleaning object, allowing method chaining.
        """
        self.cleaning_steps.append(cleaning_step)
        return self
    
    def clean_text(self, text: str) -> str:
        """
        Apply the cleaning steps to the input text.

        This method iterates over all the cleaning steps in the order they were added, applying each one
        to the text. The logger tracks each step as it is applied.

        Parameters:
        - text (str): The input text to be cleaned.

        Returns:
        - str: The cleaned text after all cleaning steps have been applied.
        """
        pages_str = None
        for cleaning_step in self.cleaning_steps:
            self.logger.debug("Performing cleaning step: %s", cleaning_step.__name__)
            pages_str = cleaning_step(text)
        
        return pages_str