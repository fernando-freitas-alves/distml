from abc import ABC
from typing import Optional

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchnet.dataset import SplitDataset


class Dataset(ABC):
    path: str
    num_partitions: int
    batch_size: int
    base_dataset: Optional[TorchDataset]
    dataset: TorchDataset

    def __init__(
        self,
        path: str,
        num_partitions: int,
        batch_size: Optional[int] = None,
        base_dataset: Optional[TorchDataset] = None,
    ) -> None:
        if not base_dataset:
            raise Exception("Argument 'base_dataset' must be defined")

        self.path = path
        self.num_partitions = num_partitions
        self.batch_size = batch_size or int(128 / num_partitions)
        self.partitions = {i: 1 / num_partitions for i in range(num_partitions)}
        self.__get_dataset(base_dataset)

    def __get_dataset(self, base_dataset: TorchDataset) -> None:
        if self.num_partitions > 1:
            self.dataset = SplitDataset(base_dataset, partitions=self.partitions)
        else:
            self.dataset = base_dataset

    def get_train_set(self, partition: int, shuffle: bool = True) -> DataLoader:
        if self.num_partitions > 1:
            self.dataset.select(partition)
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)
