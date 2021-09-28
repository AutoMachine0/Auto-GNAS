import torch
from sklearn.metrics import f1_score

class Evaluator:
    """
    Realizing the f1 score metric

    Args:
        y_predict: tensor
            the output of downstream task model
        y_ture: tensor
            the output labels for y_predict

    Returns:
        f1 score: float
            the f1 score performance
    """

    def function(self, y_predict, y_ture):
        _, y_predict = torch.max(y_predict, dim=1)
        y_predict = y_predict.to("cpu").numpy()
        y_ture = y_ture.to("cpu").numpy()
        f1_value = f1_score(y_ture, y_predict, average="macro")
        return f1_value