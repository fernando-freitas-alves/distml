from abc import ABC
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchnet.dataset import SplitDataset

DEFAULT_PATH = r"./data"


class Dataset(ABC):
    dataset: TorchDataset

    def __init__(
        self,
        num_partitions: int,
        batch_size: Optional[int] = None,
        path: Optional[str] = DEFAULT_PATH,
        base_dataset: Optional[TorchDataset] = None,
    ) -> None:
        assert base_dataset is not None

        self.num_partitions = num_partitions
        self.path = path
        self.batch_size = batch_size or int(128 / num_partitions)
        self.partitions = {i: 1 / num_partitions for i in range(num_partitions)}
        self.__load_dataset(base_dataset)

    def __load_dataset(self, base_dataset: TorchDataset) -> None:
        self.dataset = SplitDataset(base_dataset, partitions=self.partitions)
        # self.dataset = DataPartitioner(base_dataset, partitions=self.partitions)

    def train_set(self, partition: int, shuffle: bool = True) -> DataLoader:
        self.dataset.select(partition)  # type: ignore[attr-defined]
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)
        # dataset = self.dataset.use(partition)  # type: ignore[attr-defined]
        # return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


# from random import Random


# class Partition(object):
#     """ Dataset-like object, but only access a subset of it. """

#     def __init__(self, data, index):
#         self.data = data
#         self.index = index

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]


# class DataPartitioner(object):
#     """ Partitions a dataset into different chunks. """

#     def __init__(self, data, partitions=dict, seed=1234):
#         self.data = data
#         self.partitions = []

#         rng = Random()
#         rng.seed(seed)
#         data_len = len(data)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)

#         for frac in partitions.values():
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]

#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])
