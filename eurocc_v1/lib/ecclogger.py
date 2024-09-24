import logging

from eurocc_v1.paths import LOGS_DIR

def setup_logger():
    LOGS_DIR.mkdir(exist_ok=True)

    logger = logging.getLogger('ecclogger')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)s')

    fh = logging.FileHandler(LOGS_DIR / 'eurocc.log')
    ch = logging.StreamHandler()

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

setup_logger()