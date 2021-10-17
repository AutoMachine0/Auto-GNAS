## Optimizer function user-defined specification

- Users only need to define their own optimizer function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **/autognas/model/optimizer_function/**. the AutoGNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python

# import what you need to import python package

class Optimizer:
    """
    Realizing the user-defined optimizer object
    
    Args:
       gnn_model: model object
            the pytorch model object
       optimizer_parameter_dict: dict
            the hyper parameter for optimizer
    Returns:
       optimizer: optimizier object
            the user-defined optimizer object
    """

    def __init__(self,
                 gnn_model,
                 optimizer_parameter_dict):

         # User-defined  area

    def function(self):
        
         # User-defined  area
        
        return optimizer
```

