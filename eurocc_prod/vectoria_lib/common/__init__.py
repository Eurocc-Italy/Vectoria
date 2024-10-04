from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger

logger = setup_logger(
    'common', 
    Config().get("vectoria_logs_dir") / "common.log",
    Config().get("log_level")
)

logger.debug("Module common initialized")
