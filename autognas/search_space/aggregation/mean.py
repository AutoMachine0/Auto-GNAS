import torch.nn
from torch_scatter import scatter_mean

class Aggregation(torch.nn.Module):
    """
    Realizing mean pooling aggregation manner for the source_node_representation_with_coefficient
    by default PYG mean pooling function

    Args:
       heads: int
          the number of multi heads
       source_node_representation_with_coefficient:tensor
          the source node representation matrix with attention coefficient
           source_node_representation_with_coefficient = attention_coefficient * x_j
       edge_index: tensor
          the corresponding relationship between source node number
          and target node number, edge_index = [edge_index_j,edge_index_i]

    Returns:
       node_representation: none
          the node representation after mean pooling aggregating
    """

    def __init__(self):
        super(Aggregation, self).__init__()
        pass

    def function(self,
                 source_node_representation_with_coefficient,
                 edge_index):
        node_representation_agg_based_on_edge_target = scatter_mean(source_node_representation_with_coefficient,
                                                                    edge_index[1],
                                                                    dim=0)
        return node_representation_agg_based_on_edge_target