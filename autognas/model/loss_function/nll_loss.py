import torch.nn.functional as F

class Loss:
    """
    Realizing the loss object

    Args:
        predict_y: tensor
            the predict y of downstream task model

        true_y: tensor
            the true y of downstream

    Returns:
        loss: tensor
            the loss tensor variable that can calculate gradient for pytorch
    """

    def function(self, predict_y, true_y):
        loss_function = F.nll_loss
        loss = loss_function(predict_y, true_y)
        return loss