import os
import yaml
import logging
from pathlib import Path

from vectoria_lib.common.utils import Singleton
from vectoria_lib.common.paths import ETC_DIR
from vectoria_lib.common.logger import setup_logger


class Config(metaclass=Singleton):

    def __init__(self, config_path: Path | str = None):
        self.config_stream_logger = setup_logger('config_logger', 'DEBUG') # <- stream logger
        self.logger = logging.getLogger('common')
        self.config = {}
        self.load_config(config_path)
        self._langchain_tracking()
        self._disable_ragas_tracking()
        self._disable_tokenizer_parallelism()

    def load_config(self, config_path: Path | str):
        self.config_stream_logger.debug("Loading configuration from %s", config_path)
        if config_path is None:
            config_path = ETC_DIR / "default" / "default_config.yaml"
            self.config_stream_logger.info("Loading default configuration: %s", config_path)
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.config["vectoria_logs_dir"] = Path(self.config["vectoria_logs_dir"]).expanduser()
        self.config["vectoria_logs_dir"].mkdir(parents=True, exist_ok=True)
        
        self.config_stream_logger.debug("Configuration loaded: %s", self.config)
        return self        
    
    def update_from_args(self, args):
        """Override config parameters from CLI arguments"""
        for key, value in vars(args).items():
            if value is not None:
                self.config[key] = value

    def get(self, key, subkey=None):
        if subkey is None:
            return self.config[key]
        else:
            return self.config[key][subkey]

    def set(self, key, subkey=None, value=None):
        if value is None:
            self.logger.warning("Value is None for key: %s, subkey: %s", key, subkey)
            return
        if subkey is None:
            if key not in self.config:
                raise ValueError(f"Key {key} not found in config")
            self.config[key] = value
        else:
            if key not in self.config:
                raise ValueError(f"Key {key} not found in config")
            if subkey not in self.config[key]:
                raise ValueError(f"Subkey {subkey} not found in key {key} of config")
            self.config[key][subkey] = value
        if key == "langchain_tracking":
            self._langchain_tracking()



    def _langchain_tracking(self):
        if not self.config.get("langchain_tracking"):
            if "LANGCHAIN_TRACING_V2" in os.environ:
                self.config_stream_logger.warning("Disabling langchain tracking")
                del os.environ['LANGCHAIN_TRACING_V2']
            return

        if "LANGCHAIN_API_KEY" not in os.environ:
            raise ValueError("langchain_tracking is enabled but LANGCHAIN_API_KEY environment variable is not set")

        self.config_stream_logger.info("Langchain tracking enabled")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "vectoria" # TODO: make my dynamic and fetch the version

    def _disable_ragas_tracking(self):
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"

    def _disable_tokenizer_parallelism(self):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
