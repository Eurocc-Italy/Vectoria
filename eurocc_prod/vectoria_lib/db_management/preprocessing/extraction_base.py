#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

from pathlib import Path
from abc import ABC, abstractmethod

class ExtractionBase(ABC):
    # @abstractmethod
    # def extract_text_from_folder(self, folder_path: Path, limit: int = -1) -> list[str]:
    #     pass

    """
    Abstract base class for text extraction operations.

    """

    @abstractmethod
    def extract_text_from_file(self, file_path: Path) -> str:
        """
        Abstract method to extract text from a single file.

        Parameters:
        - file_path (Path): The path to the file from which text should be extracted.

        Returns:
        - str: The extracted text from the file.
        """
        pass
