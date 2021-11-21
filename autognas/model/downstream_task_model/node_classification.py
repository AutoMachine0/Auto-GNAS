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
        output_dim = graph_data.num_labels
        self.mlp = torch.nn.Linear(gnn_embedding_dim, output_dim)

    def forward(self,
                node_embedding_matrix,
                batch_train_x_index,
                mode="train"):

        logits = self.mlp(node_embedding_matrix)
        predict_y = F.log_softmax(logits, 1)

        if mode == "train":
            predict_y = predict_y[self.graph_data.train_mask]
        elif mode == "val":
            predict_y = predict_y[self.graph_data.val_mask]
        elif mode == "test":
            predict_y = predict_y[self.graph_data.test_mask]
        else:
            print("wrong mode")
            raise
        return predict_y