import torch
from torch import nn


class FFNet(nn.Module):
    """Simple neural net class.

    Goal is not to be comprehensive but rather to provide a simple interface to a toy network that allows to quickly scale the size of the network. 
    """

    def __init__(self,
                 input_size: int = 100,
                 num_hl: int = 0,
                 hl_size: int = 10):
        """
        Args:
            input_size (int): dimensionality of the input. Default is 100.
            num_hl (int): Number of hidden layers (layers between input and output layer) in the network. Default is 0.
            hl_size (int): Number of units per hidden layer. Default is 10.
        """
        super(FFNet, self).__init__()
        # initial fully connected layer that ingests input
        self._fc_in = nn.Linear(input_size, hl_size)
        self._relu = nn.ReLU()

        # intermediate hidden layers, easily allow us to grow the network
        self._hidden_layers = None
        if num_hl > 0:
            hls = []
            for _ in range(num_hl):
                hls += [nn.Linear(hl_size, hl_size), nn.ReLU()]
            self._hidden_layers = nn.Sequential(*hls)

        # output layer, no activation function for simplicity
        self._fc_out = nn.Linear(hl_size, 1)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        """Forward pass of data through network.

        Args:
            x (torch.Tensor): Input data. Shape must be Bxinput_size, where B is the batch size and input_size is the data dimensionality.

        Returns:
            torch.Tensor: Inputs after being processed by the network. Shape will be Bx1 where 1 is the width of the output layer.
        """
        z = self._fc_in(x)
        z = self._relu(z)
        if self._hidden_layers:
            z = self._hidden_layers(z)
        y = self._fc_out(z)
        return y
