import os
import torch
import numpy as np
from autognas.util import Batch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit

class DATA(object):

    def __init__(self):

        self.name = ["cora_lp", "citeseer_lp", "pubmed_lp"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1):

        if dataset == "cora_lp":
            dataset = "cora"
        elif dataset == "citeseer_lp":
            dataset = "citeseer"
        elif dataset == "pubmed_lp":
            dataset = "pubmed"

        data_name = dataset
        path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/CITE/"
        dataset = Planetoid(path, dataset)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        data_ = dataset[0]
        x = data_.x
        edge_index = data_.edge_index

        data = dataset[0]

        if train_splits is not None and val_splits is not None:
            transform = RandomLinkSplit(is_undirected=True,
                                        num_val=val_splits,
                                        num_test=1-train_splits-val_splits)

        else:
            transform = RandomLinkSplit(is_undirected=True,)

        train_data, val_data, test_data = transform(data)

        pos_train_edge_index = train_data.edge_label_index.tolist()
        neg_train_edge_index = negative_sampling(
            edge_index=train_data.edge_label_index,
            num_nodes=data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            force_undirected=True).tolist()

        # construct labels
        train_labels = self.get_link_labels(pos_train_edge_index,
                                            neg_train_edge_index)

        self.train_x = [x.to(device)]
        self.val_x = [x.to(device)]
        self.test_x = [x.to(device)]

        self.train_edge_index = [edge_index.to(device)]
        self.val_edge_index = [edge_index.to(device)]
        self.test_edge_index = [edge_index.to(device)]

        self.train_y = [torch.tensor(train_labels, dtype=torch.float32).to(device)]

        self.train_pos_edge_index = torch.tensor(pos_train_edge_index,
                                                 dtype=torch.int64).to(device)

        self.val_edge = val_data.edge_label_index.to(device)
        self.test_edge = test_data.edge_label_index.to(device)

        self.val_y = [val_data.edge_label.to(device)]
        self.test_y = [test_data.edge_label.to(device)]

        # batch process
        self.batch_train_x_list, \
        self.batch_train_edge_index_list, \
        self.batch_train_y_list, \
        self.batch_train_x_index_list = Batch(self.train_x,
                                              self.train_edge_index,
                                              self.train_y,
                                              train_batch_size).data

        self.batch_val_x_list, \
        self.batch_val_edge_index_list, \
        self.batch_val_y_list, \
        self.batch_val_x_index_list = Batch(self.val_x,
                                            self.val_edge_index,
                                            self.val_y,
                                            val_batch_size).data

        self.batch_test_x_list, \
        self.batch_test_edge_index_list, \
        self.batch_test_y_list, \
        self.batch_test_x_index_list = Batch(self.test_x,
                                             self.test_edge_index,
                                             self.test_y,
                                             test_batch_size).data

        self.num_features = data.num_features
        self.num_labels = 2
        self.data_name = data_name
        self.num_nodes = data_.num_nodes

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = len(pos_edge_index[0]) + len(neg_edge_index[0])
        link_labels = np.array([0 for i in range(num_links)])
        link_labels[:len(pos_edge_index[0])] = 1.
        return link_labels

if __name__=="__main__":

    graph = DATA()
    graph.get_data("cora_lp",
                   train_splits=0.85,
                   val_splits=0.05,
                   shuffle_flag=True)
    pass