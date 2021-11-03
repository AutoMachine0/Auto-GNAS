import torch
import torch.nn.functional as F

class Activation:
    """
    Realizing the elu activation function object

    Args:
        none

    Returns:
        act: activation object
            the elu activation function object
    """

    def function(self):
        act = torch.nn.functional.elu
        return act