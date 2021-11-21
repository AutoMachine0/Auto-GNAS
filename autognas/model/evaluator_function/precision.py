import torch
import numpy as np
import warnings
warnings.filterwarnings('always')  # "ignore" for ignoring all warning
from sklearn.metrics import precision_score

class Evaluator:
    """
    Realizing the precision metric

    Args:
        y_predict: tensor
            the output of downstream task model
        y_ture: tensor
            the output labels for y_predict

    Returns:
        precision: float
            the precision performance
    """

    def function(self, y_predict, y_ture):
        _, y_predict = torch.max(y_predict, dim=1)
        y_predict = y_predict.to("cpu").detach().numpy()
        y_ture = y_ture.to("cpu").detach().numpy()
        precision = precision_score(y_ture, y_predict, average='macro', labels=np.unique(y_predict))
        return precision