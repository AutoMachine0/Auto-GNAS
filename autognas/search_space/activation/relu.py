import torch
import torch.nn.functional as F

class Activation:
    """
    Realizing the relu activation function object

    Args:
        none

    Returns:
        act: activation object
            the relu activation function object
    """

    def function(self):
        act = torch.nn.functional.relu
        return act