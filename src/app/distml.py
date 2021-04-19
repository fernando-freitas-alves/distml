from os import environ, getenv
from typing import Dict, List, Optional

import torch.distributed as dist
from torch.multiprocessing import Manager, Process

from app.dataset import Dataset
from app.model import Model
from app.training import DistTraining
from app.training.support import TrainingResults
from app.utils.path import join_paths

OUTPUT_FILENAMES = {
    "neural_network": "trained_model.pkl",
    "training_results": "training_results.csv",
}

# TODO: #2 (requires #1) bring world_results to DistML
# FIXME: #3 Manager() is not working
# world_results: Dict[int, TrainingResults] = Manager().dict()
world_results: Dict[int, TrainingResults] = {}


class DistML:
    world_size: int
    dataset: Dataset
    neural_network: Model
    training: DistTraining
    num_epochs: int
    output_folder: Optional[str]
    verbose: bool
    processes: List[Process]
    backend: str

    def __init__(
        self,
        world_size: int,
        dataset: Dataset,
        neural_network: Model,
        training: DistTraining,
        num_epochs: int,
        verbose: bool = False,
        backend: str = "gloo",
    ) -> None:
        self.world_size = world_size
        self.dataset = dataset
        self.neural_network = neural_network
        self.training = training
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.backend = backend
        self.__config()

    def __config(self) -> List[Process]:
        environ["MASTER_ADDR"] = getenv("MASTER_ADDR") or "127.0.0.1"
        environ["MASTER_PORT"] = getenv("MASTER_PORT") or "29500"

        self.processes = []
        for rank in range(self.world_size):
            process = Process(
                target=process_exec,
                args=(
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
            process.start()
            self.processes.append(process)

        return self.processes

    def __combine_results(
        self, world_results: Dict[int, TrainingResults]
    ) -> TrainingResults:
        # FIXME: #4 world_results is not being shared among the workers
        for rank, training_results in world_results.items():
            print(f"Rank {rank}: {training_results}")

        return TrainingResults()

    def start(self) -> None:
        for process in self.processes:
            process.join()

    def save(self, output_folder: str) -> None:
        output_filepaths = join_paths(
            folderpath=output_folder, filenames=OUTPUT_FILENAMES
        )
        training_results = self.__combine_results(world_results)

        training_results.save(output_filepaths["training_results"])
        self.neural_network.save(output_filepaths["neural_network"])


# TODO: #1 make pickle works with process_exec
def process_exec(
    rank: int,
    world_size: int,
    dataset: Dataset,
    neural_network: Model,
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
