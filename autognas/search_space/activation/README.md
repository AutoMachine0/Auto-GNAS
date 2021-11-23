## Activation function user-defined specification

- Users only need to define their own attention function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/search_space/activation/**. the AutoGNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
# import what you need to import python package

class Activation:
    """
    Realizing the user-defined activation function object

    Args:
        none

    Returns:
        act: activation object
            the user-defined activation function object
    """

    def function(self):
        
        # User-defined  area
        
        return act
```

