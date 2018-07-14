import logging


class FileLogger:
    def __init__(self, file_path, level=logging.DEBUG):
        self.log_file = file_path
        logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s',
                            datefmt='%m/%d-%H:%M:%S',
                            filename=self.log_file,
                            level=level)

    def debug(self, msg):
        print(msg)
        logging.debug(msg)

    def debug_(self, msg):
        print(msg)

    def info(self, msg):
        print(msg)
        logging.info(msg)

    def info_(self, msg):
        print(msg)

    def warning(self, msg):
        print(msg)
        logging.warning(msg)

    def error(self, msg):
        print(msg)
        logging.error(msg)
