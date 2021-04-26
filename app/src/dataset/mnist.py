from functools import partial
from typing import Optional

from torch.utils.data import Dataset as TorchDataset
from torchvision import datasets, transforms
from utils.logging import getLogger

from . import Dataset

logger = getLogger(__name__)


class MNIST(Dataset):
    def __init__(
        self,
        path: str,
        num_partitions: int,
        batch_size: Optional[int] = None,
    ) -> None:
        super().__init__(
            path=path,
            num_partitions=num_partitions,
            batch_size=batch_size,
            base_dataset=self._get_base_dataset(
                path=path, transform=self._build_transform()
            ),
        )

    @staticmethod
    def _build_transform() -> transforms.Compose:
        return transforms.Compose(
            (
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            )
        )

    @classmethod
    def _get_base_dataset(
        cls, path: str, transform: transforms.Compose
    ) -> TorchDataset:
        load_method = partial(
            datasets.MNIST, root=path, train=True, transform=transform
        )

        try:
            logger.debug(f"Trying to load cached {cls.__name__} dataset...")
            base_dataset = load_method(download=False)
        except RuntimeError:
            logger.debug(
                f"Cache not found! Downloading {cls.__name__} dataset"
                " (this may take a while)..."
            )
            base_dataset = load_method(download=True)

        logger.debug(f"{cls.__name__} dataset successfully loaded")
        return base_dataset
