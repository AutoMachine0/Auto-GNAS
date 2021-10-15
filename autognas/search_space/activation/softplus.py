import torch
import torch.nn.functional as F

class Activation:
    """
   Realizing the softplus activation function object

   Args:
       none

   Returns:
       act: activation object
           the softplus activation function object
   """

    def function(self):
        act = torch.nn.functional.softplus
        return act