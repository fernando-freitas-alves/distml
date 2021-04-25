from typing import List


class Epoch(object):
    num: int
    rank: int
    num_batches: int
    losses: List[float]

    def __init__(self, num: int, *, rank: int, num_batches: int) -> None:
        self.num = num
        self.rank = rank
        self.num_batches = num_batches
        self.losses = []

    @property
    def cost(self) -> float:
        return sum(self.losses) / self.num_batches

    def __str__(self) -> str:
        return f"Epoch {self.num}, Rank {self.rank}: cost = {self.cost}"

    def save_csv(file):
        pass  # TODO: #6 (requires #4) save epoch if sharing issue between nodes is solved  # noqa: E501
