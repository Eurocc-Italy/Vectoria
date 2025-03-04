from pathlib import Path
from vectoria_lib.tasks.inference import inference
from vectoria_lib.common.paths import TEST_DIR

def test_inference(config, index_test_folder, clear_inference_engine_cache):
    
    config.set("inference_engine", "max_new_tokens", value=300)
    config.set("inference_engine", "load_in_4bit", value=True)
    config.set("inference_engine", "load_in_8bit", value=False)
    config.set("inference_engine", "device_map", value=None)


    inference_config = dict(
        index_path = index_test_folder,
        test_set_path = TEST_DIR / "data" / "eval" / "q.json",
        output_dir = Path(config.get("vectoria_logs_dir")) / "inference_results"
    )

    inference(**inference_config)