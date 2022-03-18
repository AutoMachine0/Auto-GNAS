import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add, scatter_max

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
            1.batch_train_x_list
            2.batch_train_edge_index_list
            3.batch_train_y_list
            4.batch_train_x_index_list
            5.batch_val_x_list
            6.batch_val_edge_index_list
            7.batch_val_y_list
            8.batch_val_x_index_list
            9.batch_test_x_list
            10.batch_test_edge_index_list
            11.batch_test_y_list
            12.batch_test_x_index_list
            13. num_features
            14.num_labels
            15.data_name
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

        # batch_graph_embedding = self.node_embedding_sum_pooling_to_graph_embedding(logits,
        #                                                                       batch_train_x_index)

        batch_graph_embedding = self.node_embedding_sum_pooling_to_graph_embedding(node_embedding_matrix,
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

    def node_embedding_mean_pooling_to_graph_embedding(self,
                                                       batch_node_embedding_matrix,
                                                       index):

        graph_embedding = scatter_mean(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding

    def node_embedding_sum_pooling_to_graph_embedding(self,
                                                      batch_node_embedding_matrix,
                                                      index):

        graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding

    def node_embedding_max_pooling_to_graph_embedding(self,
                                                      batch_node_embedding_matrix,
                                                      index):

        graph_embedding, _ = scatter_max(batch_node_embedding_matrix, index, dim=0)
        return graph_embedding


