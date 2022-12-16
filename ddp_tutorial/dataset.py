import torch
from typing import Tuple


class ToyDataset:
    """Simple dataset class.

    Creates artificial data, goal is to easily stream data without dataloading being a bottleneck. 
    Should be passed to torch.utils.data.DataLoader.
    """

    def __init__(self,
                 input_size: int = 100,
                 num_samples: int = 1_024):
        """
        Args:
            input_size (int): Dimensionality of the data
            num_samples (int): Simulated size of the dataset
        """
        self._input_size = input_size
        self._num_samples = num_samples

    def __len__(self) -> int:
        """
        Returns:
            int: Number of samples in the dataset
        """
        return self._num_samples

    def __getitem__(self,
                    index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            index (int): Ignored since data is randomly generated on the fly.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: sample, label data pair
        """
        return torch.randn(self._input_size), torch.randn(1)
