import torch

class Activation:
    """
    Realizing the sigmoid activation function object

    Args:
        none

    Returns:
        act: activation object
            the sigmoid activation function object
    """

    def function(self):
        act = torch.sigmoid
        return act