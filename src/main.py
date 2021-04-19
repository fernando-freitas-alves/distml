#!/usr/bin/env python

from os import path
from typing import Dict, Optional, Type

from torch.multiprocessing import freeze_support

import cli
from app import DistML
from app.dataset import MNIST, Dataset, datasets
from app.model import Model, TestNet, models
from app.training import DistTraining, SyncedSGD, training_methods
from app.training.support import TrainingResults
from app.utils.logger import getLogger

logger = getLogger(__name__)


def main(
    world_size: int,
    dataset_class: Type[Dataset],
    model_class: Type[Model],
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
    kwargs = {
        "world_size": cli.FLAGS.world_size,
        "dataset_class": datasets.get(cli.FLAGS.dataset),
        "model_class": models.get(cli.FLAGS.model),
        "training_class": training_methods.get(cli.FLAGS.training),
        "num_epochs": cli.FLAGS.epochs,
        "verbose": cli.FLAGS.verbose,
        "output_folder": cli.FLAGS.output,
    }

    title = cli.title(**kwargs)
    logger.info(title)  # FIXME: logger printing on stdout, but it should follow config
    if kwargs["verbose"]:
        print(title)

    main(**kwargs)


if __name__ == "__main__":
    cli.run(cli_entry_point)
