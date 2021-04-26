import csv
import datetime
import io
import os.path
from logging import LogRecord  # noqa: F401
from logging import FileHandler, Formatter, StreamHandler, basicConfig, root
from sys import stdout

from .path import mkdir


class CsvFormatter(Formatter):
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


def config_logger(level: int, filename: str) -> None:
    folderpath = os.path.dirname(filename)
    mkdir(folderpath)

    basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-9s  %(name)-26s:%(lineno)-5d  %(message)s",
        handlers=[
            FileHandler(filename, mode="a"),
            StreamHandler(stdout),
        ],
    )
    root.handlers[0].setFormatter(CsvFormatter())
