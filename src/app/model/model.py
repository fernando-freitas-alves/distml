import pickle
from abc import ABC

from app.utils.path import save_object
from torch.nn import Module


class Model(ABC, Module):
    def save(self, filepath: str) -> None:
        save_object(obj=self, filepath=filepath)
