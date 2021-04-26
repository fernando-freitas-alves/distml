from abc import ABC

from torch.nn import Module
from utils.path import save_object


class NeuralNetwork(ABC, Module):
    def save(self, filepath: str) -> None:
        save_object(obj=self, filepath=filepath)
