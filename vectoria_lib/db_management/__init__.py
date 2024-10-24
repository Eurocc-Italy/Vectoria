from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger
logger = setup_logger(
    'db_management', 
    Config().get("log_level"),
    Config().get("vectoria_logs_dir") / "db_management.log"
)
logger.debug("Module db_management initialized")