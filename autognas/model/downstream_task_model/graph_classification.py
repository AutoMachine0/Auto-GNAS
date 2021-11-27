import torch
import torch.nn.functional as F
from autognas.model.util import node_embedding_mean_pooling_to_graph_embedding, \
                                node_embedding_sum_pooling_to_graph_embedding, \
                                node_embedding_max_pooling_to_graph_embedding

class DownstreamTask(torch.nn.Module):
    """
    The custom downstream task class,
    using the mlp to realize the inductive graph
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

    def __init__(self,
                 gnn_embedding_dim,
                 graph_data):
        super(DownstreamTask, self).__init__()
        self.graph_data = graph_data
        self.output_dim = graph_data.num_labels
        self.mlp = torch.nn.Linear(gnn_embedding_dim, self.output_dim)

    def forward(self,
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        #logits = self.mlp(node_embedding_matrix)

        # batch_graph_embedding = node_embedding_sum_pooling_to_graph_embedding(logits,
        #                                                                       batch_train_x_index)

        batch_graph_embedding = node_embedding_sum_pooling_to_graph_embedding(node_embedding_matrix,
                                                                               batch_x_index)

        logits = self.mlp(batch_graph_embedding)

        if mode == "train":
            predict_y = F.log_softmax(logits, 1)


        elif mode == "val":
            predict_y = F.log_softmax(logits, 1)


        elif mode == "test":
            predict_y = F.log_softmax(logits, 1)


        else:
            print("wrong mode")
            raise

        return predict_y
