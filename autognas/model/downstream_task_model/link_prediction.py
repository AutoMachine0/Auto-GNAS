import torch
import torch.nn.functional as F
import random

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
        node_embedding: tensor
            the output node embedding matrix of stack gcn model

    Returns:
        predict_y: tensor
            the output tensor of mlp for transductive node
            classification
    """

    def __init__(self, gnn_embedding_dim, graph_data):
        super(DownstreamTask, self).__init__()
        self.graph_data = graph_data

    def forward(self,
                node_embedding_matrix,
                batch_train_x_index,
                mode="train"):

        if mode == "train":
            pos_edge_index = self.graph_data.train_pos_edge_index
            neg_edge_index = random.choice(self.graph_data.train_neg_edge_index_list)

        elif mode == "val":
            pos_edge_index = self.graph_data.val_pos_edge_index
            neg_edge_index = self.graph_data.val_neg_edge_index

        elif mode == "test":
            pos_edge_index = self.graph_data.test_pos_edge_index
            neg_edge_index = self.graph_data.test_neg_edge_index
        else:
            print("wrong mode")
            raise

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        source_node_embedding = node_embedding_matrix[edge_index[0]]
        target_node_embedding = node_embedding_matrix[edge_index[1]]
        link_predict = (source_node_embedding * target_node_embedding).sum(dim=-1)
        link_predict = link_predict.sigmoid()

        return link_predict