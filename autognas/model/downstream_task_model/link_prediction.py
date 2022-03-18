import torch
from torch_geometric.utils import negative_sampling

class DownstreamTask(torch.nn.Module):
    """
    The custom downstream task class,
    using the mlp to realize the inductive link
    prediction based on node embedding from
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

    def __init__(self, gnn_embedding_dim, graph_data):
        super(DownstreamTask, self).__init__()
        self.graph_data = graph_data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,
                node_embedding_matrix,
                batch_x_index,
                mode="train"):

        if mode == "train":
            pos_edge_index = self.graph_data.train_pos_edge_index
            neg_edge_index = negative_sampling(edge_index=self.graph_data.train_pos_edge_index,
                                              # num_nodes=self.graph_data.num_nodes,
                                               num_neg_samples=self.graph_data.train_pos_edge_index.size(1),
                                               force_undirected=True).to(self.device)
            edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)

        elif mode == "val":

            edge_index = self.graph_data.val_edge

        elif mode == "test":

            edge_index = self.graph_data.test_edge
        else:
            print("wrong mode")
            raise



        source_node_embedding = node_embedding_matrix[edge_index[0]]
        target_node_embedding = node_embedding_matrix[edge_index[1]]
        link_predict = (source_node_embedding * target_node_embedding).sum(dim=-1)
        link_predict = link_predict.sigmoid()

        return link_predict
