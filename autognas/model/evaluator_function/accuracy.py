import torch
import warnings
warnings.filterwarnings('always')  # "ignore" for ignoring all warning
from sklearn.metrics import accuracy_score

class Evaluator:
    """
    Realizing the accuracy metric

    Args:
        y_predict: tensor
            the output of downstream task model
        y_ture: tensor
            the output labels for y_predict

    Returns:
        accuracy: float
            the accuracy performance
    """

    def function(self, y_predict, y_ture):
        #print(y_predict)
        _, y_predict = torch.max(y_predict, dim=1)
        #print(y_predict)

        y_predict = y_predict.to("cpu").detach().numpy()
        y_ture = y_ture.to("cpu").detach().numpy()
        accuracy = accuracy_score(y_ture, y_predict)
        return accuracy