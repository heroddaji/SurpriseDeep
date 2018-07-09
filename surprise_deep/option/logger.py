import logging


class FileLogger:
    def __init__(self, file_path):
        self.log_file = file_path
        logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',
                            datefmt='%m/%d-%H:%M:%S',
                            filename=self.log_file,
                            level=logging.DEBUG)

    def debug(self, msg):
        print(msg)
        logging.debug(msg)

    def info(self, msg):
        print(msg)
        logging.info(msg)

    def warning(self, msg):
        print(msg)
        logging.warning(msg)

    def error(self, msg):
        print(msg)
        logging.error(msg)
