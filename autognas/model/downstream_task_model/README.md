## downstream task model user-defined specification

- Users only need to define their own downstream task model according to the following template in the **user-defined area** , and then put the user-defined script into this path: **autognas/model/downstream_task_model**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python
# import what you need to import python package
import torch
import torch.nn.functional as F

class DownstreamTask(torch.nn.Module):
    """
    The custom downstream task class,
    using the mlp to realize the transductive node
    classification based on node embedding from
    stack gcn model
    Args:
        gnn_embedding_dim: int
            the input node embedding dimension
        graph_data: graph data obj
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        node_embedding_matrix: tensor
            the output node embedding matrix of stack gcn model
        batch_x_index: tensor
            the node embedding matrix index for each graph
    Returns:
        predict_y: tensor
            the output tensor of predicting 
    """

    def __init__(self, gnn_embedding_dim, graph_data):
        super(DownstreamTask, self).__init__()
        
        # User-defined  area

    def forward(self,
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        # User-defined  area

        if mode == "train":
             # User-defined  area
        elif mode == "val":
             # User-defined  area
        elif mode == "test":
             # User-defined  area
        else:
            print("wrong mode")
            raise
            
        return predict_y
```
