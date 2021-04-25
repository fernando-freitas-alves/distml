from abc import ABC
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torchnet.dataset import SplitDataset

DEFAULT_PATH = r"./data"


class Dataset(ABC):
    num_partitions: int
    batch_size: int
    base_dataset: Optional[TorchDataset]
    dataset: TorchDataset

    def __init__(
        self,
        num_partitions: int,
        batch_size: Optional[int] = None,
        base_dataset: Optional[TorchDataset] = None,
    ) -> None:
        assert base_dataset is not None

        self.num_partitions = num_partitions
        self.batch_size = batch_size or int(128 / num_partitions)
        self.partitions = {i: 1 / num_partitions for i in range(num_partitions)}
        self.__load_dataset(base_dataset)

    def __load_dataset(self, base_dataset: TorchDataset) -> None:
        if self.num_partitions > 1:
            self.dataset = SplitDataset(base_dataset, partitions=self.partitions)
            # self.dataset = DataPartitioner(base_dataset, partitions=self.partitions)
        else:
            self.dataset = base_dataset

    def train_set(self, partition: int, shuffle: bool = True) -> DataLoader:
        if self.num_partitions > 1:
            self.dataset.select(partition)  # type: ignore[attr-defined]
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)
        # dataset = self.dataset.use(partition)  # type: ignore[attr-defined]
        # return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
