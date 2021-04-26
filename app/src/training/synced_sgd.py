from math import ceil

import torch
import torch.distributed as dist
import torch.optim as optim
from dataset import Dataset
from neural_network import NeuralNetwork
from torch.autograd import Variable
from torch.nn.functional import nll_loss
from torch.optim.optimizer import Optimizer
from utils.logging import getLogger

from . import DEFAULT_NUM_EPOCHS, DistTraining
from .epoch import Epoch
from .results import TrainingResults

logger = getLogger(__name__)


class SyncedSGD(DistTraining):
    def __init__(self) -> None:
        self.loss_fn = nll_loss

    def train(
        self,
        rank: int,
        world_size: int,
        dataset: Dataset,
        neural_network: NeuralNetwork,
        num_epochs: int = DEFAULT_NUM_EPOCHS,
        verbose: bool = False,
    ) -> TrainingResults:
        torch.manual_seed(1234)
        optimizer = optim.SGD(neural_network.parameters(), lr=0.01, momentum=0.5)

        train_set = dataset.get_train_set(partition=rank)
        num_batches = ceil(
            len(train_set.dataset)  # type: ignore[arg-type]
            / float(train_set.batch_size or 1)
        )

        results = TrainingResults()
        for num in range(num_epochs):
            epoch = Epoch(num=num, rank=rank, num_batches=num_batches)

            for data, target in train_set:
                loss = self.__step(
                    world_size=world_size,
                    neural_network=neural_network,
                    optimizer=optimizer,
                    data=Variable(data),
                    target=Variable(target),
                )
                epoch.losses.append(loss)

            results.append(epoch=epoch)

            logger.debug(epoch)
            if verbose:
                print(epoch)

        return results

    def __step(
        self,
        world_size: int,
        neural_network: NeuralNetwork,
        optimizer: Optimizer,
        data: Variable,
        target: Variable,
    ) -> float:
        optimizer.zero_grad()

        prediction = neural_network(data)
        loss = self.loss_fn(prediction, target)
        loss.backward()
        self.__average_gradients(world_size=world_size, neural_network=neural_network)

        optimizer.step()

        return loss.item()

    @staticmethod
    def __average_gradients(world_size: int, neural_network: NeuralNetwork) -> None:
        for param in neural_network.parameters():
            dist.all_reduce(tensor=param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size
