import torch
import copy
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops
#from autognas.model.stack_gcn_encoder.message_passing import MessagePassing
from torch_geometric.nn import MessagePassing
from autognas.dynamic_configuration import attention_getter, aggregation_getter

from torch_geometric.data import Data  # for unit test

class MessagePassingNet(MessagePassing):
    """
    The message passing network model parameters initializing,
    realizing message passing process including following process:
    1. removing the every node self loop in the input graph data
    2. adding the node self loop for the input graph data again
    3. transformer the input node feature dimension
    4. computing the attention correlation coefficient between node i and j
    5. the attention correlation coefficient multiple the feature matrix
    6. aggregating the feature matrix with the attention correlation coefficient
       for every central node i
    7. concat or average the multi head output features.

    Args:
        input_dim: int
            the input feature dimension
        output_dim: int
            the output feature dimension
        heads: int
            the number of multi heads
        concat: bool
            controlling the output feature whether need concat operator
        dropout: float
            the drop out rate for feature matrix with the attention
            correlation coefficient
        bias: bool
           controlling the output feature whether need bias operator
        att_type: str
            the attention function type for computing attention
            correlation coefficient
        agg_type: str
            the aggregation function type for node aggregation operator

    Returns:
        node_representation: tensor
            the output representation matrix
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 heads=1,
                 concat=True,
                 dropout=0,
                 bias=True,
                 att_type="gcn",
                 agg_type="sum"):

        self.custom_agg_type = False

        if agg_type in ["mean", "max"]:
            super(MessagePassingNet, self).__init__(agg_type)
        else:
            if agg_type == "sum":
                super(MessagePassingNet, self).__init__('add')
            else:
                super(MessagePassingNet, self).__init__('add')
                self.custom_agg_type = True
                self.aggregation_dict = aggregation_getter()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.att_type = att_type
        self.agg_type = agg_type
        self.bias = bias

        self.weight = Parameter(torch.Tensor(self.heads,
                                             self.input_dim,
                                             self.output_dim))
        glorot(self.weight)

        if self.bias and concat:
            self.bias = Parameter(torch.Tensor(self.heads * self.output_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(self.output_dim))
        else:
            self.bias = None

        if self.bias is not None:
            zeros(self.bias)

        self.attention_dict = attention_getter(self.heads, self.output_dim)

    def forward(self, x, edge_index):

        edge_index, _ = remove_self_loops(edge_index)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        node_nums = x.shape[0]

        for weight_index in range(self.weight.shape[0]):

            if weight_index == 0:
                x_ = torch.mm(x, self.weight[weight_index])
                edge_index_ = copy.deepcopy(edge_index)
            else:
                edge_index = edge_index + node_nums
                x_ = torch.cat([x_, torch.mm(x, self.weight[weight_index])], dim=0)
                edge_index_ = torch.cat([edge_index_, edge_index], dim=-1)

        return self.propagate(edge_index_, x=x_, num_nodes=x.size(0))

    def message(self, x_i, x_j, edge_index, num_nodes):

        self.edge_index = edge_index

        attention_function = self.attention_dict[self.att_type]

        attention_coefficient = attention_function.function(x_i,
                                                            x_j,
                                                            edge_index,
                                                            num_nodes)

        self.source_node_representation_with_coefficent = attention_coefficient * x_j

        if self.training and self.dropout > 0:

            self.source_node_representation_with_coefficent = F.dropout(self.source_node_representation_with_coefficent,
                                                                        p=self.dropout,
                                                                        training=True)

        if self.custom_agg_type:

            return self.custom_agg_update(), self.custom_agg_type

        return self.source_node_representation_with_coefficent #, self.custom_agg_type

    def update(self, aggr_out):

        node_representation = aggr_out

        return self.node_representation_transformer(node_representation)

    def custom_agg_update(self):

        aggregation_function = self.aggregation_dict[self.agg_type]

        node_representation = aggregation_function.function(self.heads,
                                                            self.source_node_representation_with_coefficent,
                                                            self.edge_index)

        return self.node_representation_transformer(node_representation)

    def node_representation_transformer(self, node_representation_):

        node_representation_ = node_representation_.view(self.heads,
                                                         int(node_representation_.shape[0]/self.heads),
                                                         self.output_dim)

        if self.concat is True:
            for index in range(self.heads):
                if index == 0:
                    node_representation = node_representation_[index]
                else:
                    node_representation = torch.cat([node_representation,
                                                    node_representation_[index]],
                                                    dim=1)
        else:
            for index in range(self.heads):
                if index == 0:
                    node_representation = node_representation_[index]
                else:
                    node_representation = node_representation + node_representation_[index]

            node_representation = node_representation / self.heads

        if self.bias is not None:
            node_representation = node_representation + self.bias

        return node_representation

# unit test
if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edges = [[0, 0, 0, 1, 2, 2, 3, 3], [1, 2, 3, 0, 0, 3, 0, 2]]

    edge_index = torch.tensor(edges, dtype=torch.long).to(device)

    node_features = [[-1, 1, 2], [1, 1, 1], [0, 1, 2], [3, 1, 2]]
    x = torch.tensor(node_features, dtype=torch.float).to(device)

    data = Data(x=x, edge_index=edge_index)

    x = data.x
    edge_index = data.edge_index

    GNN = MessagePassingNet(3, 5).to(device)

    y = GNN(x, edge_index)

    print("input:", x)
    print("output:", y)