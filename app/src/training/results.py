from os import linesep
from typing import List

from pandas import DataFrame
from utils.path import mkdir

from .epoch import Epoch


class TrainingResults(object):
    epochs: List[Epoch]

    def __init__(self) -> None:
        self.epochs = []

    def __str__(self) -> str:
        out = ""
        for epoch in self.epochs:
            out += str(epoch) + linesep
        return out

    def append(self, epoch: Epoch) -> "TrainingResults":
        self.epochs.append(epoch)
        return self

    def save(self, filepath: str) -> None:
        df = DataFrame(columns=("num", "rank", "num_batches", "cost"))

        for epoch in self.epochs:
            row = {
                "num": epoch.num,
                "rank": epoch.rank,
                "num_batches": epoch.num_batches,
                "cost": epoch.cost,
            }
            df = df.append(row, ignore_index=True)

        mkdir(filepath)
        df.to_csv(filepath)
