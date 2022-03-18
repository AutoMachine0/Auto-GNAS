import os
import torch
import random
import numpy as np
from autognas.util import Batch
from torch_geometric.datasets import Planetoid

class DATA(object):

    def __init__(self):

        self.name = ["cora", "citeseer", "pubmed"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123,
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1):

        data_name = dataset
        path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/CITE/" + dataset
        dataset = Planetoid(path, dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset[0]

        edge_index = data.edge_index.to(device)
        x = data.x.to(device)
        y = data.y.to(device)

        index_list = [i for i in range(y.size(0))]

        # construct transductive node classification task mask
        if shuffle_flag:

            if not random_seed:
                random_seed = 123
            random.seed(random_seed)

            random.shuffle(index_list)

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits+val_splits]
                idx_test = index_list[train_splits+val_splits:train_splits+val_splits+test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]
        else:

            if train_splits == None or val_splits == None:

                train_splits = self.count_(data.train_mask)
                val_splits = self.count_(data.val_mask)
                test_splits = self.count_(data.test_mask)

                idx_train = index_list[:train_splits]
                idx_val = index_list[train_splits:train_splits + val_splits]
                idx_test = index_list[train_splits + val_splits:train_splits + val_splits + test_splits]

            else:

                idx_train = index_list[:int(y.size(0) * train_splits)]
                idx_val = index_list[int(y.size(0) * train_splits):int(y.size(0) * train_splits) + int(y.size(0) * val_splits)]
                idx_test = index_list[int(y.size(0) * train_splits) + int(y.size(0) * val_splits):]

        self.train_mask = torch.tensor(self.sample_mask_(idx_train, y.size(0)), dtype=torch.bool)
        self.val_mask = torch.tensor(self.sample_mask_(idx_val, y.size(0)), dtype=torch.bool)
        self.test_mask = torch.tensor(self.sample_mask_(idx_test, y.size(0)), dtype=torch.bool)

        # Auto-GNAS input required attribution
        self.train_y = [y[self.train_mask]]
        self.val_y = [y[self.val_mask]]
        self.test_y = [y[self.test_mask]]

        self.train_x = [x]
        self.val_x = [x]
        self.test_x = [x]

        self.train_edge_index = [edge_index]
        self.val_edge_index = [edge_index]
        self.test_edge_index = [edge_index]

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
        self.num_labels = y.max().item() + 1
        self.data_name = data_name

    def sample_mask_(self, idx, l):
        """ create mask """
        mask = np.zeros(l)
        for index in idx:
            mask[index] = 1
        return np.array(mask, dtype=np.int32)

    def count_(self, mask):
        true_num = 0
        for i in mask:
            if i:
                true_num += 1
        return true_num

if __name__=="__main__":

    graph = DATA()
    graph.get_data("cora", shuffle_flag=True)