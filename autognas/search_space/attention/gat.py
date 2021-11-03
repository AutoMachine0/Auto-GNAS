import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot

class Attention(torch.nn.Module):
    """
    Computing the gat attention correlation coefficient
    for each node of input graph data set

    Args:
        heads: int
           the number of multi heads
        output_dim: int
           the transformer dimension of input in this stack gcn layer
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

    def __init__(self,
                 heads,
                 output_dim):

        super(Attention, self).__init__()
        self.heads = heads
        self.output_dim = output_dim
        self.a_list = []
        self.a = Parameter(torch.Tensor(self.heads, 1, self.output_dim*2))
        glorot(self.a)

    def function(self,
                 x_i,
                 x_j,
                 edge_index,
                 num_nodes):

        e_ = torch.cat([x_i, x_j], dim=1).view(self.heads,
                                               int(x_i.shape[0]/self.heads),
                                               self.output_dim*2)

        for index in range(self.heads):
            if index == 0:
                e = F.leaky_relu(torch.mm(e_[index], self.a[index].t()), 0.2)
            else:
                e = torch.cat([e,
                              F.leaky_relu(torch.mm(e_[index], self.a[index].t()), 0.2)],
                              dim=0)

        edge_index_i = edge_index[1]

        attention_coefficient = softmax(src=e,
                                        index=edge_index_i,
                                        num_nodes=num_nodes*self.heads)

        return attention_coefficient



