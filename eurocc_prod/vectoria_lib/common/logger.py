import logging

def setup_logger(logger_name: str, log_file: str = None, log_level: str = None):
    """
    Sets up a logger with the given name.
    Optionally logs to a file if log_file is provided.
    """
    logger = logging.getLogger(logger_name)
    if log_level is not None:
        logger.setLevel(logging.getLevelName(log_level))
    else:
        logger.setLevel(logging.DEBUG)  # Set default logging level

    # Define a formatter (use same format for all loggers)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(filename)s - %(lineno)s')

    # Stream handler (console logging)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (if log_file is provided)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
