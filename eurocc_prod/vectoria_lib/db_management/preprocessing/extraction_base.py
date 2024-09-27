#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@dataclass
class DocumentData:
    unstructured_text: str = ""
    structured_text: dict = field(default_factory=dict)  # Use default_factory for mutable types
    metadata: dict = field(default_factory=dict)         # Same for metadata


class ExtractionBase(ABC):

    @abstractmethod
    def extract_text_from_file(self, file_path: Path) -> DocumentData:
        """
        Abstract method to extract text from a single file.

        Parameters:
        - file_path (Path): The path to the file from which text should be extracted.

        Returns:
        - DocumentData: The extracted data from the file.
        """
        pass
