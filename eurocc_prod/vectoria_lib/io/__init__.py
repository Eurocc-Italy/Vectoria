from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger
logger = setup_logger(
    'io', 
    Config().get("vectoria_logs_dir") / "io.log"
)
logger.info("Module io initialized")