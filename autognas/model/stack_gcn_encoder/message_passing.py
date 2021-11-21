import torch
from autognas.dynamic_configuration import aggregation_getter


class MessagePassing(torch.nn.Module):

    def __init__(self, agg_type="sum"):
        super(MessagePassing, self).__init__()
        self.agg_type = agg_type
        self.aggregation_dict = aggregation_getter()

    def propagate(self, edge_index, x, num_nodes):

        x_i, x_j = self.node_feature_expand_based_on_index(x, edge_index)

        source_node_representation_with_coefficent = self.message(x_i, x_j, edge_index, num_nodes)

        aggregation_function = self.aggregation_dict[self.agg_type]

        node_representation = aggregation_function.function(source_node_representation_with_coefficent,
                                                            edge_index)

        out = self.update(node_representation)

        return out

    def message(self, x_i, x_j, edge_index, num_nodes):

        pass

    def update(self, out):

        return out

    def node_feature_expand_based_on_index(self, x, edge_index):
        target_node_edge_index = edge_index[1]
        source_node_edge_index = edge_index[0]
        x_i = x[target_node_edge_index]
        x_j = x[source_node_edge_index]
        return x_i, x_j

if __name__ == "__main__":
    pass