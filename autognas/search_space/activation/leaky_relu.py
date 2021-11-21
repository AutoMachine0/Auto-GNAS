import torch
import torch.nn.functional as F

class Activation:
    """
    Realizing the leaky_relu activation function object

    Args:
        none

    Returns:
        act: activation object
            the leaky_relu activation function object
    """

    def function(self):
        act = torch.nn.functional.leaky_relu
        return act