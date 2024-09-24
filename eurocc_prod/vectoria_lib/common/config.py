import yaml
import logging
from pathlib import Path

from vectoria_lib.common.utils import Singleton
from vectoria_lib.common.paths import ETC_DIR
class Config(metaclass=Singleton):

    """
    TODO: do this!
    @staticmethod
    def load_config(config_path: Path | str = None):
        return Config(config_path)
    """
    def __init__(self, config_path: Path | str = None):
        self.logger = logging.getLogger('common')
        self.config = {}
        self.load_config(config_path)

    def load_config(self, config_path: Path | str = None):
        self.logger.debug("Loading configuration from %s", config_path)

        if config_path is None:
            self.logger.debug("No configuration path provided, using default")
            config_path = ETC_DIR / "vectoria_config/default_config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)

        self.config["vectoria_logs_dir"] = Path(self.config["vectoria_logs_dir"]).expanduser()
        self.config["vectoria_logs_dir"].mkdir(parents=True, exist_ok=True)
        
        self.logger.debug("Configuration loaded: %s", self.config)

    def get(self, key):
        return self.config.get(key)

    def set(self, key, value):
        self.config[key] = value
