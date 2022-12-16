import os
import argparse
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.multiprocessing as mp
from model import FFNet
from dataset import ToyDataset


def run_training(gpu: int,
                 args: dict):
    """Main training code

    Args:
        gpu (int): index of the gpu for this process on this process' node
        args (dict): dictionary of args
    """
    # By manually seeding we ensure all models start from the same initial state
    # Another way of accomplishing this is by having one process (e.g. rank 0) broadcast its model at the start
    torch.manual_seed(0)
    print(f'Process index: {gpu}')

    batch_size = 8
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='gloo',
        world_size=args.world_size,
        rank=rank
    )

    # Initialize the model on the GPU
    torch.cuda.set_device(gpu)
    nn_mdl = FFNet(input_size=args.input_size,
                   num_hl=args.num_hl, hl_size=args.hl_size)
    nn_mdl.cuda(gpu)

    # Optimizer and loss (loss has to be put on GPU too)
    optimizer = optim.SGD(nn_mdl.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.MSELoss().cuda(gpu)

    # Distributing the model is as easy as that
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])

    # Initialize the dataset
    train_ds = ToyDataset(input_size=args.input_size,
                          num_samples=args.num_samples)

    # This will communicate to the process which subset of the dataset to draw from
    # Essentially the partition helper from the naive example
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds,
        num_replicas=args.world_size,
        rank=rank
    )

    # torch dataloader interfaces nicely with the distributed sampler
    train_set = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler)

    for epoch in range(args.epochs):
        running_loss = 0.
        for samples, labels in train_set:
            # data has to be loaded to the gpu
            samples.cuda(non_blocking=True)
            labels.cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = nn_mdl(samples)
            loss = loss_fn(outputs, labels)
            loss.backward()
            # at this point we previously had to average the gradients
            # torch native DDP uses hooks on the autograd graph to do this in the backend
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        if args.verbose:
            print(
                f'Process {rank} / {args.world_size} -> Epoch: {epoch + 1} - total loss: {running_loss / 2000:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, help='number of nodes')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-i', "--input_size", type=int, default=100,
                        help="dimensionality of the data")
    parser.add_argument('-nb', "--num_samples", type=int, default=1_024,
                        help="size of the dataset")
    parser.add_argument('-d', "--num_hl", type=int, default=0,
                        help="number of hidden layers in the network")
    parser.add_argument('-z', "--hl_size", type=int, default=100,
                        help="size of the hidden layers in the network")
    parser.add_argument('-e', "--epochs", type=int, default=10,
                        help="number of epochs to train for")
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Print training process. 1=verbose, 0=silent.')

    args = parser.parse_args()

    # one process per GPU, nodes could be scaled too but not on the UH cluster since multi-node-gpu jobs are prohibited
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8899'
    mp.spawn(run_training, nprocs=args.gpus, args=(args,))
