import logging

def setup_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] - <%(name)s> - [%(levelname)s] - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt=formatter)
    logger.addHandler(hdlr=stream_handler)
    return logger