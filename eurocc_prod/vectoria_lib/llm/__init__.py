from vectoria_lib.common.config import Config
from vectoria_lib.common.logger import setup_logger

logger = setup_logger(
    'llm', 
    Config().get("vectoria_logs_dir") / "llm.log"
)

logger.info("Module llm initialized")
