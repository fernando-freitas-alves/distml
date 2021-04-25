from abc import ABC, abstractmethod
from typing import Optional

from dataset import Dataset
from neural_network import NeuralNetwork

from .results import TrainingResults

DEFAULT_NUM_EPOCHS = 10


class DistTraining(ABC):
    @abstractmethod
    def train(
        self,
        rank: int,
        world_size: int,
        dataset: Dataset,
        neural_network: NeuralNetwork,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        verbose: bool = False,
    ) -> TrainingResults:
        ...
