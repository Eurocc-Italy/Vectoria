# Vectoria: Supercomputer-driven Vector Database for Private LLM Retrieval

Vectoria is an advanced, AI-driven solution for optimizing organizational knowledge access through private large language model (LLM) retrieval. It enables seamless interaction with internal documentation via chatbot-like interfaces while maintaining high standards of data privacy and precision.

## Key Features
- **Efficient Knowledge Management**: Unified access to scattered internal documentation through a centralized query system.
- **High Precision and Relevance**: Leverages vector-based retrieval techniques for precise query matching.
- **Data Privacy**: Operates entirely within the organization’s infrastructure, ensuring compliance with internal security protocols.

---

## Architectural Overview

Vectoria’s architecture consists of tightly integrated components to deliver exceptional performance and accuracy. Below are the core modules and workflows:

### Core Modules:
1. **Document Preprocessor**: Ingests, cleans, and splits internal documents into manageable text chunks.
2. **Vector Database**: Stores vector embeddings of document chunks for efficient similarity searches.
3. **Retriever**: Locates relevant chunks from the vector database based on user queries.
4. **Generative Model**: Synthesizes human-like answers using a pre-trained large language model.

### Operational Tasks:
1. **Build Vector Database**:
   - Transforms internal documents into a searchable structure (referred to as "Build Index").
   - Supported formats: PDF, DOCX.
2. **Inference**:
   - Combines retrieval and generative AI to provide context-aware responses.

---

## Workflows

### Build Vector Database

**Steps:**
1. **Data Preprocessing**: Cleans and splits documents into chunks (e.g., 512 characters).
2. **Data Embedding**: Generates vector embeddings using pre-trained models (e.g., `bge-m3`).
3. **Data Storing**: Stores embeddings in a vector database (e.g., FAISS), attaching metadata for efficient filtering and explainability.

**Example Command:**
```bash
python vectoria \
  --config 'etc/default/default_config.yaml' \
  build_index \
  --input-docs-dir 'test/data' \
  --output-dir 'test/index' \
  --output-suffix '_my_test_index'
