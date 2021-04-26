from logging import getLevelName

import absl.logging
from absl import flags
from absl.app import run  # noqa: F401
from absl.flags import FLAGS  # noqa: F401
from constants import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_FOLDER,
    DEFAULT_EPOCHS,
    DEFAULT_LOGGING_LEVEL,
    DEFAULT_MASTER_ADDR,
    DEFAULT_MASTER_PORT,
    DEFAULT_NEURAL_NETWORK,
    DEFAULT_NUM_NODES,
    DEFAULT_OUTPUT,
    DEFAULT_TRAINING,
    DEFAULT_VERBOSE,
    DEFAULT_WORKERS_PER_NODE,
    UNDEFINED_NODE_ID,
)
from utils.logging import root

flags.DEFINE_integer(
    "node_id",
    short_name="id",
    help=f"Node ID [default: {UNDEFINED_NODE_ID}]",
    default=None,
    lower_bound=0,
)

flags.DEFINE_integer(
    "num_workers",
    short_name="w",
    help=f"Num of local workers (in this node) [default: {DEFAULT_WORKERS_PER_NODE}]",
    default=None,
    lower_bound=1,
)

flags.DEFINE_integer(
    "num_nodes",
    short_name="n",
    help=f"Num of nodes [default: {DEFAULT_NUM_NODES}]",
    default=None,
    lower_bound=1,
)

flags.DEFINE_string(
    "dataset",
    short_name="d",
    help=f"Dataset class name [default: {DEFAULT_DATASET}]",
    default=None,
)

flags.DEFINE_string(
    "neural_network",
    short_name="nn",
    help=f"Neural network model class name [default: {DEFAULT_NEURAL_NETWORK}]",
    default=None,
)

flags.DEFINE_string(
    "training",
    short_name="t",
    help=f"Distribution training class name [default: {DEFAULT_TRAINING}]",
    default=None,
)

flags.DEFINE_integer(
    "epochs",
    short_name="e",
    help=f"Num of epochs during training [default: {DEFAULT_EPOCHS}]",
    default=None,
    lower_bound=1,
)

flags.DEFINE_string(
    "master_addr",
    short_name="ip",
    help=f"Master node IP address [default: {DEFAULT_MASTER_ADDR}]",
    default=None,
)

flags.DEFINE_string(
    "master_port",
    short_name="p",
    help=f"Master node TCP port [default: {DEFAULT_MASTER_PORT}]",
    default=None,
)

flags.DEFINE_boolean(
    "verbose",
    short_name="vv",
    help=f"Print some details do stdout [default: {DEFAULT_VERBOSE}]",
    default=None,
)

flags.DEFINE_string(
    "logging_level",
    short_name="l",
    help=(
        "Logging level"
        " ('DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL')"
        f" [default: {getLevelName(DEFAULT_LOGGING_LEVEL)}]"
    ),
    default=None,
)

flags.DEFINE_string(
    "dataset_folder",
    short_name="df",
    help=(
        "Output folder path where the dataset will be saved"
        f" [default: {DEFAULT_DATASET_FOLDER}]"
    ),
    default=None,
)

flags.DEFINE_string(
    "output",
    short_name="o",
    help=(
        "Output folder path where the training results will be saved"
        f" [default: {DEFAULT_OUTPUT}]"
    ),
    default=None,
)


def remove_absl_handler():
    root.removeHandler(absl.logging._absl_handler)
