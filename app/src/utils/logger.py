import csv
import datetime
import io
import logging
from logging import LogRecord, getLogger

from .path import mkdir


class CsvFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self.output = io.StringIO()
        self.writer = csv.writer(self.output, quoting=csv.QUOTE_ALL)

    def format(self, record: LogRecord) -> str:
        record.asctime = datetime.datetime.now().isoformat()

        self.writer.writerow(
            [record.asctime, record.levelname, record.name, record.msg]
        )

        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)

        return data.strip()


mkdir("output")

logging.basicConfig(
    filename="output/main.log",
    filemode="a",
    level=logging.DEBUG,
)

logging.root.handlers[0].setFormatter(CsvFormatter())
