## Attention function user-defined specification

- Users only need to define their own attention function according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/search_space/attention/**. the AutoGNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
import torch
# import what you need to import python package

class Attention(torch.nn.Module):
    """
    Computing the attention correlation coefficient
    for each node of input graph data set
    Args:
        heads: int
           the number of multi heads
        output_dim: int
           the transformer dimension of input in this gnn layer
        x_i: tenser
           the extended node feature matrix based on edge_index_i
           the edge_index_i is the target node number list
        x_j: tensor
           the extended node feature matrix based on edge_index_j
           the edge_index_j is the source node number list
        edge_index: tensor
           the corresponding relationship between source node number
           and target node number, edge_index = [edge_index_j,edge_index_i]
        num_nodes: int
           the number of node in the input graph data
    Returns:
        attention_coefficient: tensor
           the gat attention correlation coefficient for x_j node feature matrix
    """

    def __init__(self, heads, output_dim):

        super(Attention, self).__init__()
        
        # User-defined  area
        
    def function(self, x_i, x_j, edge_index, num_nodes):
        
        # User-defined area

        return attention_coefficient
```

