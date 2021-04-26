from os import environ
from typing import Dict, List, Optional

import torch.distributed as dist
from constants import OUTPUT_FILENAMES
from dataset import Dataset
from neural_network import NeuralNetwork
from torch.multiprocessing import Process
from training import DistTraining
from training.results import TrainingResults
from utils.logging import getLogger
from utils.path import join_paths

logger = getLogger(__name__)

# TODO: #2 (requires #1) bring world_results to DistML
# FIXME: #3 Manager() is not working
# from torch.multiprocessing import Manager
# world_results: Dict[int, TrainingResults] = Manager().dict()
world_results: Dict[int, TrainingResults] = {}


# TODO: #7 simplify classes definition with attrs (https://www.attrs.org/)
class DistML:
    node_id: int
    num_workers: int
    world_size: int
    rank_range: List[int]
    dataset: Dataset
    neural_network: NeuralNetwork
    training: DistTraining
    num_epochs: int
    master_addr: str
    master_port: str
    verbose: bool
    output_folder: Optional[str]
    processes: List[Process]
    backend: str

    def __init__(
        self,
        node_id: int,
        num_workers: int,
        world_size: int,
        dataset: Dataset,
        neural_network: NeuralNetwork,
        training: DistTraining,
        num_epochs: int,
        master_addr: str,
        master_port: str,
        verbose: bool = False,
        backend: str = "gloo",
    ) -> None:
        self.node_id = node_id
        self.num_workers = num_workers
        self.world_size = world_size
        self.dataset = dataset
        self.neural_network = neural_network
        self.training = training
        self.num_epochs = num_epochs
        self.master_addr = master_addr
        self.master_port = master_port
        self.verbose = verbose
        self.backend = backend
        self.__config()

    def __build_rank_range(self) -> None:
        rank_range_start = self.node_id * self.num_workers
        rank_range_end = (self.node_id + 1) * self.num_workers

        if rank_range_end > self.world_size:
            raise Exception(
                f"Rank max of {rank_range_end} cannot surpass"
                f" world size of {self.world_size}"
            )

        self.rank_range = list(range(rank_range_start, rank_range_end))

    def __config_master_protocol(self) -> None:
        # Torch distributed communication package
        # https://pytorch.org/docs/stable/distributed.html
        environ["MASTER_ADDR"] = self.master_addr
        environ["MASTER_PORT"] = self.master_port

    def __config(self) -> List[Process]:
        logger.debug(
            f"Configuring {self.num_workers} local workers /"
            f" {self.world_size} world size on node #{self.node_id}..."
        )

        self.__config_master_protocol()
        self.__build_rank_range()

        self.local_workers = {}
        for rank in self.rank_range:
            process = Process(
                target=process_exec,
                args=(
                    self.node_id,
                    rank,
                    self.world_size,
                    self.dataset,
                    self.neural_network,
                    self.training,
                    self.num_epochs,
                    self.backend,
                    self.verbose,
                ),
            )
            self.local_workers[rank] = process

            logger.debug(
                f"Starting worker {rank}/{self.world_size-1} process"
                f" on node #{self.node_id}..."
            )
            process.start()
            logger.debug(
                f"Worker {rank}/{self.world_size-1} successfully started"
                f" on node #{self.node_id}..."
            )

        return self.local_workers

    def __combine_results(
        self, world_results: Dict[int, TrainingResults]
    ) -> TrainingResults:
        # FIXME: #4 world_results is not being shared among the workers, so I cannot be combined into a single output  # noqa: E501
        for rank, training_results in world_results.items():
            print(f"Rank #{rank}: {training_results}")

        return TrainingResults()

    def start(self) -> None:
        logger.debug("Starting DistML...")

        for rank, process in self.local_workers.items():
            # TODO: #8 properly use queue log handlers to manage multiprocess logging into a single file (https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes)  # noqa: E501
            logger.debug(
                f"Joining worker {rank}/{self.world_size-1} process"
                f" on node #{self.node_id}..."
            )
            process.join()
            logger.debug(
                f"Ending worker {rank}/{self.world_size-1} process"
                f" on node #{self.node_id}..."
            )

    def save(self, output_folder: str) -> None:
        output_filepaths = join_paths(
            folderpath=output_folder, filenames=OUTPUT_FILENAMES
        )
        training_results = self.__combine_results(world_results)

        training_results.save(output_filepaths["training_results"])
        self.neural_network.save(output_filepaths["neural_network"])


# TODO: #1 make pickle works with process_exec
def process_exec(
    node_id: int,
    rank: int,
    world_size: int,
    dataset: Dataset,
    neural_network: NeuralNetwork,
    training: DistTraining,
    num_epochs: int,
    backend: str,
    verbose: bool = False,
) -> None:
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    training_results: TrainingResults = training.train(
        rank=rank,
        world_size=world_size,
        dataset=dataset,
        neural_network=neural_network,
        num_epochs=num_epochs,
        verbose=verbose,
    )

    world_results[rank] = training_results

    logger.debug(
        f"Worker {rank}/{world_size -1} process finished" f" on node #{node_id}..."
    )
