import torch
import torch.nn.functional as F

class Activation:
    """
    Realizing the relu6 activation function object

    Args:
        none

    Returns:
        act: activation object
            the relu6 activation function object
    """

    def function(self):
        act = torch.nn.functional.relu6
        return act