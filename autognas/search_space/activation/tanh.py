import torch

class Activation:
    """
    Realizing the tanh activation function object

    Args:
       none

    Returns:
       act: activation object
           the tanh activation function object
    """

    def function(self):
        act = torch.tanh
        return act