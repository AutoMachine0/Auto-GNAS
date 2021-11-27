## Loss function user-defined specification

- Users only need to define their own loss function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/model/loss_function**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
# import what you need to import python package
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
        
        # User-defined  area
        
        return loss
```

