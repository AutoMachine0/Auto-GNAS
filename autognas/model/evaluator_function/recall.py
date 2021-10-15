import torch
import warnings
warnings.filterwarnings('always')  # "ignore" for ignoring all warning
from sklearn.metrics import recall_score

class Evaluator:
    """
    Realizing the recall metric

    Args:
        y_predict: tensor
            the output of downstream task model
        y_ture: tensor
            the output labels for y_predict

    Returns:
        recall: float
            the recall performance
    """

    def function(self, y_predict, y_ture):
        _, y_predict = torch.max(y_predict, dim=1)
        y_predict = y_predict.to("cpu").numpy()
        y_ture = y_ture.to("cpu").numpy()
        recall = recall_score(y_ture, y_predict, average="macro")
        return recall