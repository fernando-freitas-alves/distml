from abc import ABC, abstractmethod
from typing import Optional

from app.dataset import Dataset
from app.model import Model

from .support import TrainingResults

DEFAULT_NUM_EPOCHS = 10


class DistTraining(ABC):
    @abstractmethod
    def train(
        self,
        rank: int,
        world_size: int,
        dataset: Dataset,
        neural_network: Model,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        verbose: bool = False,
    ) -> TrainingResults:
        pass
