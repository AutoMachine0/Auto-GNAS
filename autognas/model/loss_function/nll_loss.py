import torch.nn.functional as F

class Loss:
    """
    Realizing the loss object

    Args:
        none

    Returns:
        loss: loss object
            the loss function object for model training
    """

    def function(self):
        loss = F.nll_loss
        return loss