import torch
import torch.nn.functional as F
from autognas.dynamic_configuration import activation_getter
from autognas.model.stack_gcn_encoder.message_passing_net import MessagePassingNet

from autognas.datasets.util_cite_network import CiteNetwork # for unit test

class GcnEncoder(torch.nn.Module):
    """
    Constructing the stack gcn model based on stack gcn architecture,
    realizing the stack gcn model forward process.

    Args:
        architecture: list
            the stack gcn architecture describe
            for example: ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
        original_feature_num: int
            the original input dimension for the stack gcn model
        dropout: float
            the drop out rate for stack gcn model for every layer

    Returns:
        output: tensor
            the output of the stack gcn model.
    """

    def __init__(self,
                 architecture,
                 original_feature_num,
                 dropout=0.6):

        super(GcnEncoder, self).__init__()

        self.architecture = architecture
        self.original_feateure_num = original_feature_num
        self.layer_num = int(len(self.architecture)/5)
        self.dropout = dropout
        self.activation_dict = activation_getter()
        self.gnn_layers_list = torch.nn.ModuleList()
        self.activation_list = []

        for layer in range(self.layer_num):

            if layer == 0:
                input_dim = self.original_feateure_num
            else:
                input_dim = hidden_dimension_num * multi_heads_num  # 构建第二层GNN输入维度

            attention_type = self.architecture[layer * 5 + 0]
            aggregator_type = self.architecture[layer * 5 + 1]
            multi_heads_num = int(self.architecture[layer * 5 + 2])
            hidden_dimension_num = int(self.architecture[layer * 5 + 3])
            activation_type = self.architecture[layer * 5 + 4]
            concat = True

            if layer == self.layer_num - 1 or self.layer_num == 1:
                concat = False

            self.gnn_layers_list.append(MessagePassingNet(input_dim,
                                                          hidden_dimension_num,
                                                          multi_heads_num,
                                                          concat,
                                                          dropout=self.dropout,
                                                          att_type=attention_type,
                                                          agg_type=aggregator_type))

            self.activation_list.append(self.activation_dict[activation_type])

    def forward(self, x, edge_index_all):
        output = x

        for activation, gnn_layer in zip(self.activation_list, self.gnn_layers_list):
            output = F.dropout(output, p=self.dropout, training=self.training)
            output = activation(gnn_layer(output, edge_index_all))

        return output

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = CiteNetwork("cora")

    in_feats = graph.num_features

    gnn_architecture = ['gcn', 'sum',  1, 10, 'tanh', 'gcn', 'sum', 1, 5, 'sigmoid']

    MyGNN = GcnEncoder(gnn_architecture, in_feats).to(device)

    output = MyGNN(graph.x, graph.edge_index)

    print(output)
