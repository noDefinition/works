def get_logger(log_file_path):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, mode='w')
    fh.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s|%(filename)s[line:%(lineno)d]|%(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
