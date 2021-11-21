import warnings
warnings.filterwarnings('always')  # "ignore" for ignoring all warning
from sklearn.metrics import roc_auc_score

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
        y_predict = y_predict.to("cpu").detach().numpy()
        y_ture = y_ture.to("cpu").detach().numpy()
        roc_auc_score_ = roc_auc_score(y_ture, y_predict)
        return roc_auc_score_