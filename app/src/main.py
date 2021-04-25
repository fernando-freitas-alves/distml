#!/usr/bin/env python

import sys
from os import path
from typing import Dict, Optional, Tuple, Type

from torch.multiprocessing import freeze_support

import cli
import config
from dataset import Dataset
from dataset.mnist import MNIST
from distml import DistML
from neural_network import NeuralNetwork
from neural_network.test_net import TestNet
from training import DistTraining
from training.results import TrainingResults
from training.synced_sgd import SyncedSGD
from utils.logger import getLogger
from utils.types import build_types_map

logger = getLogger(__name__)

datasets = build_types_map((MNIST,))
models = build_types_map((TestNet,))
training_methods = build_types_map((SyncedSGD,))


def main(
    world_size: int,
    dataset_class: Type[Dataset],
    model_class: Type[NeuralNetwork],
    training_class: Type[DistTraining],
    num_epochs: int,
    verbose: bool = False,
    output_folder: Optional[str] = None,
) -> None:
    dataset = dataset_class(num_partitions=world_size)
    neural_network = model_class()
    training = training_class()
    distml = DistML(
        world_size=world_size,
        dataset=dataset,
        neural_network=neural_network,
        training=training,
        num_epochs=num_epochs,
        verbose=verbose,
    )

    distml.start()

    if output_folder:
        distml.save(output_folder)


def cli_entry_point(*args):
    cli.remove_absl_handler()

    kwargs = {
        "world_size": cli.FLAGS.world_size,
        "dataset_class": datasets.get(cli.FLAGS.dataset),
        "model_class": models.get(cli.FLAGS.neural_network),
        "training_class": training_methods.get(cli.FLAGS.training),
        "num_epochs": cli.FLAGS.epochs,
        "verbose": cli.FLAGS.verbose,
        "output_folder": cli.FLAGS.output,
    }
    title = cli.title(**kwargs)

    logger.info(title)
    if kwargs["verbose"]:
        print(title)

    main(**kwargs)


if __name__ == "__main__":
    cli.run(cli_entry_point)
