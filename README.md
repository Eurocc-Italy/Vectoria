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

![Build Vector Database workflow](images/build_index_workflow.png "Build Vector Database workflow")

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
```

**Arguments:**

- `--config`: Path to the configuration file. Example: `'etc/default/default_config.yaml'`.
- `build_index`: Specifies the command to build the vector index.
- `--input-docs-dir`: Directory containing the input documents for indexing.
- `--output-dir`: Directory where the generated vector index will be saved.
- `--output-suffix`: Optional suffix appended to the output index name.

**Output:**

```bash
INFO - Running preprocessing pipeline - preprocessing_pipeline.py - 75
INFO - Found 2 files - preprocessing_pipeline.py - 77

DEBUG - Extracting text from /path/to/data/1.pdf - extraction_pdf.py - 28
DEBUG - Extracting text from /path/to/data/2.pdf - extraction_pdf.py - 28

DEBUG - Loaded 24815 characters - extraction_pdf.py - 38
DEBUG - Seeking and replacing remove_multiple_spaces ... - utils.py - 16
DEBUG - Seeking and replacing remove_bullets ... - utils.py - 16
...     ...                                        ...				
DEBUG - Processing file /path/to/data/1.pdf took 1.21 seconds - preprocessing_pipeline.py - 102

DEBUG - Loaded 27417 characters - extraction_pdf.py - 38
DEBUG - Seeking and replacing remove_multiple_spaces ... - utils.py - 16
DEBUG - Seeking and replacing remove_bullets ... - utils.py - 16
...     ...                                        ...
DEBUG - Processing file /path/to/data/2.pdf took 1.28 seconds - preprocessing_pipeline.py - 102

INFO - Total number of chunks: 208 - preprocessing_pipeline.py - 95
INFO - Created 208 documents from /path/to/data/pdf in 1.52 seconds - build_index.py - 25
INFO - Building FAISS vector store - vectore_store_builder.py - 32
INFO - Loading Embedder model: /path/to/embedder - faiss_vector_store.py - 35
```

### Inference Workflow

![Inference workflow](images/inference_workflow.png "Inference workflow")

**Steps:**

1. **Query Reception**: Accepts user questions via CLI or file input.
2. **Query Embedding**: Converts the query into a vector embedding using a pre-trained model.
3. **Vector Search**: Finds the top-k most relevant chunks using similarity metrics from the vector database.
4. **Prompt Construction**: Combines retrieved chunks with the query to provide context for the response.
5. **Answer Generation**: Uses a generative AI model (e.g., `Llama-3.1`) to produce natural language answers.

**Example Command:**

```bash
python vectoria \
  --config 'etc/default/default_config.yaml' \
  inference \
  --index-path 'test/data/index/my_new_test_index/index.pkl' \
  --test-set-path 'test/data/results/my_questions.json' \
  --questions 'Who is the CEO?' 'Who is the VP?' \
  --output-dir 'test/data/results'
```

N.B.: `--test-set-path` and `--questions` are both not formally strictly required, since vectoria is able to handle a dual mode. Using `--questions` argument will override questions from `--test-set-path` and will provide answers only through CLI, avoiding any dump on file. IMPORTANT: at least one MUST be provided.


**Arguments:**

- `--config`: Path to the configuration file. Example: `'etc/default/default_config.yaml'`.
- `inference`: Specifies the command to perform inference on the vector database.
- `--index-path`: Path to the index file containing precomputed vector embeddings.
- `--questions`: List of user questions for inference, provided as separate strings or sourced from a file.
- `--output-dir`: Directory where inference results, such as generated answers and logs, will be saved.

**Output:**

```bash
INFO - Loading default configuration: /path/to/etc/default/default_config.yaml - config.py - 26

INFO - Building FAISS vector store - vectore_store_builder.py - 32
INFO - Loading Embedder model: /path/to/embedder - faiss_vector_store.py - 35
INFO - Loading index from disk: /path/to/index/test_index - vectore_store_builder.py - 24
INFO - Deserializing FAISS index from pickle file /path/to/index/test_index - faiss_vector_store.py - 81

INFO - Building FAISS retriever - retriever_builder.py - 23
INFO - Creating retriever from vector store with kwargs: {'k': 5, 'fetch_k': 5, 'lambda_mult': 0.5} - faiss_retriever.py - 25
INFO - Creating RAG application - agent_builder.py - 64

DEBUG - Loading tokenizer took 0.32 seconds - huggingface_inference_engine.py - 41
Loading checkpoint shards:   0%|          | 0/30 [00:00<?, ?it/s]
Loading checkpoint shards:  25%|██▋       | 8/30 [00:10<00:29,  1.32s/it]
Loading checkpoint shards:  50%|█████     | 15/30 [00:20<00:20,  1.34s/it]
Loading checkpoint shards:  75%|███████▋  | 23/30 [00:30<00:09,  1.36s/it]
Loading checkpoint shards: 100%|██████████| 30/30 [00:39<00:00,  1.33s/it]
DEBUG - Loading model /path/to/model took 42.07 seconds - huggingface_inference_engine.py - 68

INFO - Questions are loaded from test set JSON: /path/to/test.json - qa.py - 86
DEBUG - Filter prefix match 'ANSWER:'! - parser.py - 19
DEBUG - Filter postfix match 'END'! - parser.py - 28

DEBUG - Question: Which is the training regime for the models? - qa.py - 57
INFO - Answer: The models were trained for a maximum of 1 day. - qa.py - 58

INFO - Time taken to answer question: 8.58 seconds - qa.py - 111
INFO - Mean time and std taken to answer questions: 8.58 seconds, 0.00 seconds - qa.py - 129
INFO - Annotated test set saved to /path/to/results/inference_results.json and took 0.01 seconds - qa.py - 140
```
