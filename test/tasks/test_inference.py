from pathlib import Path
from vectoria_lib.tasks.inference import inference
from vectoria_lib.common.paths import TEST_DIR

def test_inference(config, clear_inference_engine_cache):
    
    config.set("langchain_tracking", value=True)
    config.set("inference_engine", "max_new_tokens", value=300)
    config.set("inference_engine", "load_in_4bit", value=False)
    config.set("inference_engine", "load_in_8bit", value=False)

    inference_config = dict(
        faiss_index_path = TEST_DIR / "data" / "index" / "BAAI__bge-m3_faiss_index",
        test_set_path = TEST_DIR / "data" / "eval" / "q.json",
        output_dir = Path(config.get("vectoria_logs_dir")) / "inference_results"
    )

    inference(**inference_config)
       
