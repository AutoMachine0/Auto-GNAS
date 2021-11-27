## Evaluator Function user-defined specification

- Users only need to define their own evaluator function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/model/evaluator_function**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
# import what you need to import python package
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
        
        # User-defined  area
       
        return accuracy
```
