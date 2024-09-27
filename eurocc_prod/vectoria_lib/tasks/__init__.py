from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger

logger = setup_logger(
    'tasks', 
    Config().get("vectoria_logs_dir") / "tasks.log"
)

logger.debug("Module tasks initialized")
