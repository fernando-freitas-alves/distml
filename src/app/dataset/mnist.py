from typing import Optional

from torchvision import datasets, transforms

from .dataset import DEFAULT_PATH, Dataset


class MNIST(Dataset):
    def __init__(
        self,
        num_partitions: int,
        batch_size: Optional[int] = None,
        path: Optional[str] = DEFAULT_PATH,
    ) -> None:
        transform = transforms.Compose(
            (
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            )
        )
        base_dataset = datasets.MNIST(
            path, train=True, download=True, transform=transform
        )
        super().__init__(num_partitions, batch_size, path, base_dataset)
