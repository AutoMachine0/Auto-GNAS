import os
import torch
from autognas.util import Batch
from torch_geometric.datasets import TUDataset

class DATA(object):

    def __init__(self):
        self.name = ["PROTEINS", "MUTAG", "DHFR", "COX2", "BZR", "AIDS"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1):

        if train_splits == None or val_splits == None:
            train_splits = 0.8
            val_splits = 0.1
            shuffle_flag = True

        data_name = dataset
        data_path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/" + data_name

        if shuffle_flag:
            data_set = TUDataset(data_path, data_name).shuffle()
        else:
            data_set = TUDataset(data_path, data_name)

        len_data_set = len(data_set)
        train_index = int(train_splits * len_data_set)
        val_index = int(val_splits * len_data_set) + train_index

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_x = self.data_x_split(data_set, 0, train_index)
        self.train_y = self.data_y_split(data_set, 0, train_index)
        self.train_edge_index = self.data_edge_split(data_set, 0, train_index)

        self.val_x = self.data_x_split(data_set, train_index, val_index)
        self.val_y = self.data_y_split(data_set, train_index, val_index)
        self.val_edge_index = self.data_edge_split(data_set, train_index, val_index)

        self.test_x = self.data_x_split(data_set, val_index, len_data_set)
        self.test_y = self.data_y_split(data_set, val_index, len_data_set)
        self.test_edge_index = self.data_edge_split(data_set, val_index, len_data_set)

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

        self.num_features = data_set.num_node_features
        self.num_labels = data_set.num_classes
        self.data_name = data_name

    def data_x_split(self,
                     data_set,
                     index_start,
                     index_end):
        data_x_list = []
        for data in data_set[index_start:index_end]:
            data_x_list.append(data.x.to(self.device))
        return data_x_list

    def data_y_split(self,
                     data_set,
                     index_start,
                     index_end):
        data_y_list = []
        for data in data_set[index_start:index_end]:
            data_y_list.append(data.y.to(self.device))
        return data_y_list

    def data_edge_split(self,
                        data_set,
                        index_start,
                        index_end):
        data_edge_list = []
        for data in data_set[index_start:index_end]:
            data_edge_list.append(data.edge_index.to(self.device))
        return data_edge_list

if __name__=="__main__":
    data = DATA()
    data.get_data("MUTAG")
    pass