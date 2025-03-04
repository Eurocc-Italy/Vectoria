from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger
logger = setup_logger(
    'rag', 
    Config().get("log_level"),
    Config().get("vectoria_logs_dir") / "rag.log"
)
logger.debug("Module rag initialized")