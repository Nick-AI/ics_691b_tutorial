import os
import torch
import argparse
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from random import Random
from typing import Iterable, Optional, Callable
from model import FFNet
from dataset import ToyDataset


#############################
# Data Partitioning
#############################


class DatasetPartition(object):
    """Essentially the same as ToyDataset class.
    Implements basic functionality for torch dataset on partitions.
    """

    def __init__(self,
                 data: Iterable,
                 index: int):
        """
        Args:
            data (Iterable): some local data subset from global dataset
            index (int): some index for local data
        """
        self.data = data
        self.index = index

    def __len__(self) -> int:
        """Returns total size of this dataset

        Returns:
            int: total length
        """
        return len(self.index)

    def __getitem__(self,
                    idx: int) -> Iterable:
        """_summary_

        Args:
            idx (int): index for item to getch

        Returns:
            Iterable: data at index
        """
        local_index = self.index[idx]
        return self.data[local_index]


class DataPartitioner(object):
    """Helper class to go from global dataset to local partitions for each process.
    """

    def __init__(self,
                 data: Iterable,
                 world_size: int,
                 seed: int = 7):
        """
        Args:
            data (Iterable): Global dataset to be distributed among processes
            world_size (int): Number of processes among which to distribute the data
            seed (int, optional): Seed for random generator to ensure reproducability. Defaults to 7.
        """
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)

        # list of all data indices to shuffle for random distribution
        indices = list(range(0, len(data)))
        rng.shuffle(indices)

        # might drop modulo if not perfectly divisible for simplicity
        part_len = len(data) // world_size
        for _ in range(world_size):
            self.partitions.append(indices[:part_len])
            indices = indices[part_len:]

    def use(self, partition_idx: int) -> DatasetPartition:
        """Return partition at index partition_idx

        Args:
            partitio_idx (int): partition index

        Returns:
            DatasetPartition: Partition for process partition_index
        """
        return DatasetPartition(self.data, self.partitions[partition_idx])


def partition_dataset(world_size: int,
                      batch_size: int,
                      input_size: int = 100,
                      num_samples: int = 1_024) -> torch.utils.data.DataLoader:
    """Function actually splitting the dataset

    Args:
        world_size (int): Number of processes in the group
        batch_size (int): Batch size per process
        input_size (int, optional): Data dimensionality. Defaults to 100.
        num_samples (int, optional): Total dataset size. Defaults to 1_024.

    Returns:
        torch.utils.data.DataLoader: Dataloader for this process' subset of the total data
    """
    batch_size

    train_ds = ToyDataset(input_size=input_size, num_samples=num_samples)
    partition = DataPartitioner(train_ds, world_size)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=batch_size,
                                            shuffle=True)
    return train_set


#############################
# Distributed SGD
#############################


def reduce_gradients(model: torch.nn.Module):
    """Function to all_reduce gradients from all processes

    First gets and sums all gradients for each parameter and then averages them.
    Stores the updated gradient information directly in the model so no need to return anything.

    Args:
        model (torch.nn.Module): Model being trained
    """
    # number of total processes for averaging
    size = float(dist.get_world_size())

    # repeat this process for each individual parameter in the model
    for param in model.parameters():
        # get this parameter from each process' model copy and them sum them all
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        # divide by number of total processes to get the final gradient
        param.grad.data /= size


#############################
# Setup & Runner
#############################


def run_training(rank: int,
                 world_size: int,
                 input_size: int = 100,
                 num_samples: int = 1_024,
                 num_hl: int = 0,
                 hl_size: int = 10,
                 epochs: int = 10,
                 verbose: int = 1):
    """Main training code

    Args:
        rank (int): Process rank
        world_size (int): Total number processes in the group
        input_size (int, optional): Data dimensionality, must be 1D. Defaults to 100.
        num_samples (int, optional): Total number of samples in the dataset. Defaults to 1_024.
        num_hl (int, optional): Number of hidden layers in the NN. Defaults to 0.
        hl_size (int, optional): Size of the hidden layers in the NN. Defaults to 10.
        epochs(int, optional): Number of epochs to train for.
        vebose (int, optional): Whether to print training progress. Default is 1 (True).
    """
    # By manually seeding we ensure all models start from the same initial state
    # Another way of accomplishing this is by having one process (e.g. rank 0) broadcast its model at the start
    torch.manual_seed(0)

    batch_size = 8
    train_set = partition_dataset(
        world_size, batch_size, input_size, num_samples)

    nn_mdl = FFNet(input_size=input_size, num_hl=num_hl, hl_size=hl_size)
    optimizer = optim.SGD(nn_mdl.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        running_loss = 0.
        for samples, labels in train_set:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nn_mdl(samples)
            loss = loss_fn(outputs, labels)
            loss.backward()

            # this time we first have to average the gradients across the process group
            reduce_gradients(nn_mdl)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if verbose:
            print(
                f'Process {rank} / {world_size} -> Epoch: {epoch + 1} - total loss: {running_loss / 2000:.3f}')


def init_process(rank: int,
                 world_size: int,
                 train_fn: Callable,
                 train_fn_kwargs: Optional[dict] = None,
                 backend: Optional[str] = 'gloo'):
    """Function called to initialize individual processes

    Args:
        rank (int): Rank of this process
        world_size (int): Number of processes in the group
        train_fn (callable): Function that actually runs model training
        backend (optional, str): Backend to use. Options are gloo, mpi, nccl. Default is gloo.
    """
    try:
        val_backends = ['gloo', 'mpi', 'nccl']
        assert backend in ['gloo', 'mpi', 'nccl']
    except AssertionError:
        raise NotImplementedError(
            f'Supported backends: {val_backends} don\'t include: {backend}')
    # There are other ways than environment variables of initializing such as TCP or a shared file system for IPC
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8899'
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size)
    if train_fn_kwargs:
        train_fn(rank, world_size, **train_fn_kwargs)
    else:
        train_fn(rank, world_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', "--world_size", type=int, default=2,
                        help="number of processes to spawn")
    parser.add_argument('-i', "--input_size", type=int, default=100,
                        help="dimensionality of the data")
    parser.add_argument('-n', "--num_samples", type=int, default=1_024,
                        help="size of the dataset")
    parser.add_argument('-d', "--num_hl", type=int, default=0,
                        help="number of hidden layers in the network")
    parser.add_argument('-z', "--hl_size", type=int, default=100,
                        help="size of the hidden layers in the network")
    parser.add_argument('-e', "--epochs", type=int, default=10,
                        help="number of epochs to train for")
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Print training process. 1=verbose, 0=silent.')

    args = parser.parse_args()

    train_fn_kwargs = {
        'input_size': args.input_size,
        'num_samples': args.num_samples,
        'num_hl': args.num_hl,
        'hl_size': args.hl_size,
        'epochs': args.epochs,
        'verbose': args.verbose
    }

    processes = []
    for rank in range(args.world_size):
        p = mp.Process(target=init_process, args=(
            rank, args.world_size, run_training, train_fn_kwargs, 'gloo'))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
