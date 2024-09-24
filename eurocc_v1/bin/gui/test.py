from eurocc_v1.lib.api.v1 import (
    create_and_write_index,
    create_qa_agent
)
from eurocc_v1.paths import DATA_DIR
from pathlib import Path
input_docs_dir = DATA_DIR / "raw"
output_index_dir = Path(__file__).parent
embedder_model = "BAAI/bge-m3"
faiss_index_path = create_and_write_index(input_docs_dir, output_index_dir, embedder_model)
