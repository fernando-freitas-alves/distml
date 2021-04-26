#!/usr/bin/env python
from logging import getLevelName
from os import linesep
from os.path import abspath, join
from typing import Any, Dict, List, Optional, Type

import cli
import env
from config import config_app
from constants import LOGGING_FILENAME, MASTER_NODE_ID, UNDEFINED_NODE_ID
from dataset import Dataset
from dataset.mnist import MNIST
from distml import DistML
from neural_network import NeuralNetwork
from neural_network.test_net import TestNet
from training import DistTraining
from training.synced_sgd import SyncedSGD
from utils.logging import config_logger, getLogger
from utils.types import build_types_map, str2bool

config_app()
logger = getLogger(__name__)

datasets = build_types_map(MNIST)
models = build_types_map(TestNet)
training_methods = build_types_map(SyncedSGD)


def main(
    node_id: Optional[int],
    num_workers: int,
    num_nodes: int,
    dataset_class: Type[Dataset],
    model_class: Type[NeuralNetwork],
    training_class: Type[DistTraining],
    num_epochs: int,
    master_addr: str,
    master_port: str,
    verbose: bool,
    logging_level: int,
    dataset_folder: str,
    output_folder: Optional[str] = None,
    title: Optional[int] = None,
) -> None:
    config_logger(level=logging_level, filename=join(output_folder, LOGGING_FILENAME))

    if title:
        logger.info(title)
        if verbose:
            print(title)

    logger.info(f"Started app on node #{node_id}")

    world_size = int(num_workers) * int(num_nodes)
    dataset = dataset_class(num_partitions=world_size, path=dataset_folder)
    neural_network = model_class()
    training = training_class()
    distml = DistML(
        node_id=node_id,
        num_workers=num_workers,
        world_size=world_size,
        dataset=dataset,
        neural_network=neural_network,
        training=training,
        num_epochs=num_epochs,
        master_addr=master_addr,
        master_port=master_port,
        verbose=verbose,
    )

    distml.start()

    if output_folder:
        distml.save(output_folder)


def build_arguments_from_env(**kwargs: Dict[str, Any]) -> Dict[str, Any]:
    setting_args = {
        "node_id": (kwargs.get("node_id") or int(env.NODE_ID) if env.NODE_ID else None),
        "num_workers": kwargs.get("num_workers") or int(env.WORKERS_PER_NODE),
        "num_nodes": kwargs.get("num_nodes") or int(env.NUM_NODES),
        "dataset_class": datasets.get(kwargs.get("dataset_class") or env.DATASET),
        "model_class": models.get(kwargs.get("model_class") or env.NEURAL_NETWORK),
        "training_class": training_methods.get(
            kwargs.get("training_class") or env.TRAINING
        ),
        "num_epochs": kwargs.get("num_epochs") or int(env.EPOCHS),
        "master_addr": kwargs.get("master_addr") or env.MASTER_ADDR,
        "master_port": kwargs.get("master_port") or env.MASTER_PORT,
        "verbose": kwargs.get("verbose") or str2bool(env.VERBOSE),
        "logging_level": (
            kwargs.get("logging_level") or getLevelName(env.LOGGING_LEVEL)
        ),
        "dataset_folder": abspath(kwargs.get("dataset_folder") or env.DATASET_FOLDER),
        "output_folder": abspath(kwargs.get("output_folder") or env.OUTPUT),
    }

    if setting_args["node_id"] == UNDEFINED_NODE_ID:
        logger.warning(
            f"Node ID not defined. Setting it to master's ({MASTER_NODE_ID})."
            " This may cause undesired results. Make sure to set it before starting."
        )
        setting_args["node_id"] = MASTER_NODE_ID

    return setting_args


def cli_entry_point(*args: List[Any]) -> None:
    cli.remove_absl_handler()

    kwargs = build_arguments_from_env(
        **{
            "node_id": cli.FLAGS.node_id,
            "num_workers": cli.FLAGS.num_workers,
            "num_nodes": cli.FLAGS.num_nodes,
            "dataset_class": cli.FLAGS.dataset,
            "model_class": cli.FLAGS.neural_network,
            "training_class": cli.FLAGS.training,
            "num_epochs": cli.FLAGS.epochs,
            "verbose": cli.FLAGS.verbose,
            "logging_level": cli.FLAGS.logging_level,
            "dataset_folder": cli.FLAGS.dataset_folder,
            "output_folder": cli.FLAGS.output,
        }
    )

    title_kwargs = {**kwargs}
    title_kwargs["logging_level"] = getLevelName(title_kwargs["logging_level"])

    main(title=build_title(**title_kwargs), **kwargs)


def build_title(**kwargs) -> str:
    def sep() -> str:
        return ("#" * 65) + linesep

    out = ""

    out += sep()
    out += "Machine Learning with Message Passing Interface" + linesep
    out += sep()
    for name, value in kwargs.items():
        out += f"  • {name}: {value}" + linesep
    out += sep()

    return out


if __name__ == "__main__":
    cli.run(cli_entry_point)
