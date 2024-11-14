#
# VECTORIA
#
# @authors : Andrea Proia, Chiara Malizia, Leonardo Baroncelli
#

import time
from typing import List
from langchain.docstore.document import Document
from vectoria_lib.rag.vector_store.vectore_store_base import VectorStoreBase
from vectoria_lib.rag.postretrieval_steps.postretrieval_step_base import PostRetrievalStepBase

class FullParagraphs(PostRetrievalStepBase):

    def __init__(self, vector_store: VectorStoreBase):
        super().__init__()
        self.wrapped_vector_store = vector_store

    def post_process(self, chunks: List[Document]) -> List[Document]:
        """
        This method retrieve the full paragraph for a given set of chunks (1 or more depending on config)
        """       
        unique_paragraph_numbers: dict[str, List[str]] = self._get_unique_paragraph_number_per_doc_file_name(chunks)

        filters = self._get_filters(unique_paragraph_numbers)

        full_paragraphs: List[Document] = self._get_full_paragraphs(filters)

        return full_paragraphs

    def _get_unique_paragraph_number_per_doc_file_name(self, chunks: List[Document]) -> dict[str, List[str]]:
        """
        doc_file_name_1 : [1, 1.1, 1.1.1, 5.5]
        doc_file_name_2 : [2, 2.1, 2.1.1, 2.2]
        """
        unique_docs: set[str] = set(doc.metadata["doc_file_name"] for doc in chunks)
        
        unique_paragraph_numbers: dict[str, List[Document]] = {}

        for doc_file_name in unique_docs:
            unique_paragraphs = [doc.metadata["paragraph_number"] for doc in chunks if doc.metadata["doc_file_name"] == doc_file_name]
            unique_paragraph_numbers[doc_file_name] = list(set(unique_paragraphs))
            self.logger.debug("Unique paragraph numbers for doc_file_name %s: %s", doc_file_name, unique_paragraph_numbers[doc_file_name])
        
        return unique_paragraph_numbers
    
    def _get_filters(self, paragraph_numbers: dict[str, List[str]]) -> List[dict]:
        """
        From:
        doc_file_name_1 : [1, 1.1, 1.1.1, 5.5]
        doc_file_name_2 : [2, 2.1, 2.1.1, 2.2]

        To:
        [
            {"doc_file_name": "doc_file_name_1", "paragraph_numbers": [1, 1.1, 1.1.1, 5.5]},
            {"doc_file_name": "doc_file_name_2", "paragraph_numbers": [2, 2.1, 2.1.1, 2.2]}
        ]
        """
        filters: List[dict] = []
        for doc_file_name, paragraph_numbers in paragraph_numbers.items():
            filters.append({"doc_file_name": doc_file_name, "paragraph_number": paragraph_numbers})

        self.logger.debug("Filters: %s", filters)
        return filters

    def _get_full_paragraphs(self, filters: List[dict]) -> List[Document]:
        start_time = time.time()

        full_paragraphs_docs : List[Document] = []

        for _filter in filters: # one filter per doc_file_name
            
            filtered_chunks = self.wrapped_vector_store.index.similarity_search(
                "*", k=1000000, fetch_k=1000000, filter=_filter
            )

            self.logger.debug("Retrieved %d metadata-filtered chunks: with metadata %s", len(filtered_chunks), _filter) 

            if len(filtered_chunks) == 0:
                raise ValueError(f"No chunks found for the given filter metadata: {_filter}")

            filtered_chunks: List[List[Document]] = self._split_by_paragraph_numbers(filtered_chunks)

            for chunk_list in filtered_chunks:

                full_paragraphs_docs.append(self._build_full_paragraph_doc(chunk_list))
        
        self.logger.debug("Retrieved full paragraphs in %.2f seconds", time.time() - start_time)

        return full_paragraphs_docs

    def _split_by_paragraph_numbers(self, chunks: List[Document]) -> List[List[Document]]:
        # group all chunks by paragraph number
        # return a list of chunks with the same paragraph number
        grouped_chunks: dict[str, List[Document]] = {}
        for chunk in chunks:
            paragraph_number = chunk.metadata["paragraph_number"]
            if paragraph_number not in grouped_chunks:
                grouped_chunks[paragraph_number] = []
            grouped_chunks[paragraph_number].append(chunk)
        return list(grouped_chunks.values())

    def _build_full_paragraph_doc(self, chunk_list: List[Document]) -> Document:
        # Sort retrieved chunks to correct order
        chunk_list.sort(key=lambda doc: doc.metadata.get("seq_id", 0))

        # Join page content to re-build full paragraph
        full_paragraph: str = "".join(doc.page_content for doc in chunk_list)
        
        metadata = chunk_list[0].metadata.copy()
        del metadata["seq_id"]
        
        return Document(page_content=full_paragraph, metadata=metadata)
