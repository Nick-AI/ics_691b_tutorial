import time
import torch
from torch import nn, optim
from .dataset import ToyDataset
from typing import List


def benchmark_training(nn_mdl: nn.Module,
                       batch_size: int = 32,
                       input_size: int = 100,
                       num_samples: int = 16_384,
                       nb_epochs: int = 10,
                       repeats: int = 10,
                       verbose: bool = True) -> List[float]:
    """Function to benchmark training time for a provided model on toy data.

    Args:
        nn_mdl (nn.Module): model to be trained 
        batch_size (int, optional): Defaults to 32.
        input_size (int, optional): Size of training data samples. Defaults to 100.
        num_samples (int, optional): Number of training data samples. Defaults to 16_384.
        nb_epochs (int, optional): How many epochs to train for. Defaults to 10.
        repeats (int, optional): How many times to repeat the whole training process. Defaults to 10.
        verbose (bool, optional): Whether to print progress between each repeat. Defaults to True.

    Returns:
        List[flaot]: List of times it took for training to complete per repeat. In nanoseconds
    """
    # initialize data
    train_ds = ToyDataset(input_size=input_size, num_samples=num_samples)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, num_workers=0, shuffle=False)

    # initialize optimizer& loss
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(nn_mdl.parameters(), lr=0.001, momentum=0.9)

    # train loop
    train_times = []
    for rep_i in range(repeats):
        if verbose:
            print(f'Repeat number: {rep_i + 1} of {repeats}')
        start = time.perf_counter_ns()
        for epoch in range(nb_epochs):
            for batch_idx, data in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                samples, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = nn_mdl(samples)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
        train_times.append(time.perf_counter_ns()-start)

    return train_times
