from ..option.run_option import RunOption
import os


class Recorder(RunOption):
    filename = 'records.csv'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.filepath = os.path.join(self.get_working_dir(), self.filename)
        self.file = None

        self._create_record_file()

    def _create_or_load_option_file(self):
        pass

    def _create_record_file(self):
        with open(self.filepath, 'a') as f:
            f.write('')

    def open(self):
        self.file = open(self.filepath, 'a')

    def close(self):
        if not self.file.closed:
            self.file.close()

    def write_line(self, msg):
        if not self.file.closed:
            self.file.write(msg + '\n')
            self.file.flush()

    def flush(self):
        if not self.file.closed:
            self.file.flush()
