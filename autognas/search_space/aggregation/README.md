## Aggregation function user-defined specification

- Users only need to define their own aggregation function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/search_space/aggregation/**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
import torch.nn
# import what you need to import python package

class Aggregation(torch.nn.Module):
    """
   Realizing user-defined aggregation manner for the source_node_representation_with_coefficient

   Args:
       source_node_representation_with_coefficient:tensor
          the source node representation matrix with attention coefficient
           source_node_representation_with_coefficient = attention_coefficient * x_j
       edge_index: tensor
          the corresponding relationship between source node number
          and target node number, edge_index = [edge_index_j,edge_index_i]

   Returns:
       node_representation_agg_based_on_edge_target: tensor
          the node representation after custum aggregating
   """

    def __init__(self):
        super(Aggregation, self).__init__()
        
        # User-defined  area

    def function(self,
                 source_node_representation_with_coefficient,
                 edge_index):
        
        # User-defined  area
   
        return  node_representation_agg_based_on_edge_target
```

