import os
import torch
import configparser
import numpy as np
from autognas.auto_model import AutoModel
from autognas.parallel import ParallelConfig

class GraphData(object):
    """
    Custom graph data class and the graph data structure follows PYG.
    suggest study the PYG data structure introduction strongly:
    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html

    the data class need required attributions as follow:

    1.self.train_x: tensor
        the training set feature matrix, for example node classification task:

        [[1.0,1.0,2.0],-> one row represents one train node feature vector
        [0.0,1.0,1.0],
        [3.0,2.0,2.0]]

    2.self.val_x: tensor
        the validation set feature matrix, the data structure like self.train_x
        for node classification task

    3.self.test_x: tensor
        the test set feature matrix, the data structure like self.train_x
        for node classification task

    4.self.train_y: tensor
        the training set label vector, for example node classification task:
        [0,1,0,0] -> every element represent one train node label

    5.self.val_y: tensor
        the validation set label vector, the data structure like self.train_y
        for node classification task

    6.self.test_y: tensor
        the testing set label vector, the data structure like self.train_y
        for node classification task

    7.self.train_edge_index: tensor
        the training set edge relationship list, for example node classification task:
        [[0,2,2,4,4,4,4,4,1,1,3,7,6,5],-> source node number
        [2,0,4,2,1,7,6,5,4,3,1,4,4,4]] -> target node number

    8.self.val_edge_index: tensor
        the validation set edge relationship list, the data structure like self.train_edge_index

    9.self.test_edge_index: tensor
        the testing set edge relationship list, the data structure like self.train_edge_index

    10.self.num_features: int
        the feature dimension of feature matrix

    11.self.data_name: str
        the data set name
   """
    def __init__(self):

        node_edge_path = os.path.split(os.path.realpath(__file__))[0][:] + "/node_edge.txt"
        node_feature_path = os.path.split(os.path.realpath(__file__))[0][:] + "/node_feature.txt"
        node_label_path = os.path.split(os.path.realpath(__file__))[0][:] + "/node_label.txt"

        # construct edge
        source_node = []
        target_node = []
        with open(node_edge_path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                line = line.split(" ")
                source_node.append(int(line[0]))
                target_node.append(int(line[1]))

        edge_index = [source_node, target_node]

        # construct node feature matrix x
        x = []
        with open(node_feature_path, "r") as f:
            for line in f.readlines():
                x.append(list(eval(line)))

        # construct label matrix y
        y = []
        with open(node_label_path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                line = line.split(" ")
                y.append(int(line[1]))

        # construct transductive task mask
        y_len = len(y)
        indices = np.arange(y_len).astype('int32')
        idx_train = indices[:int(y_len * 0.5)]
        idx_val = indices[int(y_len * 0.5):int(y_len * 0.5)+int(y_len * 0.25)]
        idx_test = indices[int(y_len * 0.5)+int(y_len * 0.25):]

        train_mask = self.sample_mask(idx_train, y_len)
        val_mask = self.sample_mask(idx_val, y_len)
        test_mask = self.sample_mask(idx_test, y_len)

        # transformer data type
        device = torch.device("cuda")

        self.edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
        self.x = torch.tensor(x, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)

        self.train_mask = torch.tensor(train_mask, dtype=torch.uint8).to(device)
        self.val_mask = torch.tensor(val_mask, dtype=torch.uint8).to(device)
        self.test_mask = torch.tensor(test_mask, dtype=torch.uint8).to(device)


        # construct required attribution for custom graph data object

        self.train_y = self.y[self.train_mask]
        self.val_y = self.y[self.val_mask]
        self.test_y = self.y[self.test_mask]

        self.train_x = self.x
        self.val_x = self.x
        self.test_x = self.x

        self.train_edge_index = self.edge_index
        self.val_edge_index = self.edge_index
        self.test_edge_index = self.edge_index

        self.num_features = int(self.x.shape[1])
        self.data_name = "example_data"

    def sample_mask(self, idx, l):
        """ create mask """
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.int32)

if __name__=="__main__":

    ParallelConfig(False)
    graph = GraphData()
    config = configparser.ConfigParser()
    config_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/config/"
    configuration = "graphpas.ini"
    config.read(config_path + configuration)
    search_parameter = dict(config.items('search_parameter'))
    gnn_parameter = dict(config.items("gnn_parameter"))
    AutoModel(graph, search_parameter, gnn_parameter)
