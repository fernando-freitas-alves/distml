from os import linesep

import absl.logging
from absl import flags
from absl.app import run
from absl.flags import FLAGS

from utils.logger import logging

flags.DEFINE_integer(
    "world_size",
    help="Num of nodes",
    default=2,
    lower_bound=1,
)

flags.DEFINE_string(
    "dataset",
    short_name="d",
    help="Dataset name",
    default="MNIST",
)

flags.DEFINE_string(
    "neural_network",
    short_name="nn",
    help="Neural network model name",
    default="TestNet",
)

flags.DEFINE_string(
    "training",
    short_name="t",
    help="Distribution training method name",
    default="SyncedSGD",
)

flags.DEFINE_integer(
    "epochs",
    short_name="e",
    help="Num of epochs",
    default=10,
    lower_bound=1,
)

flags.DEFINE_boolean(
    "verbose",
    short_name="vv",
    help="Print epoch results while training",
    default=False,
)

flags.DEFINE_string(
    "output",
    short_name="o",
    help="Output folder path where the training results will be saved",
    default="output",
)


def remove_absl_handler():
    logging.root.removeHandler(absl.logging._absl_handler)


def title(**kwargs) -> str:
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
