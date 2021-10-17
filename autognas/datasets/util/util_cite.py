import os
import torch
import random
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

class DATA(object):

    def __init__(self):

        self.name = ["cora", "citeseer", "pubmed"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123):

        data_name = dataset
        path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/CITE/" + dataset
        dataset = Planetoid(path, dataset, T.NormalizeFeatures())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset[0]

        if shuffle_flag:

            if not random_seed:
                random_seed = 123
            random.seed(random_seed)

            edge_index = data.edge_index
            x = data.x.tolist()
            y = data.y.tolist()
            index_list = [i for i in range(len(y))]
            random.shuffle(index_list)
            shuffle_x = []
            shuffle_y = []

            for index in index_list:
                shuffle_x.append(x[index])
                shuffle_y.append(y[index])

            x = torch.tensor(np.array(shuffle_x), dtype=torch.float32)
            y = torch.tensor(np.array(shuffle_y), dtype=torch.int64)

        else:
            edge_index = data.edge_index
            x = data.x
            y = data.y

        edge_index = edge_index.to(device)
        x = x.to(device)
        y = y.to(device)

        # construct transductive node classification task mask
        if train_splits == None or val_splits == None:

            self.train_mask = data.train_mask.to(device)
            self.val_mask = data.val_mask.to(device)
            self.test_mask = data.test_mask.to(device)
        else:
            y_len = len(y)

            indices = np.arange(y_len).astype('int32')
            idx_train = indices[:int(y_len * train_splits)]
            idx_val = indices[int(y_len * train_splits):int(y_len * train_splits) + int(y_len * val_splits)]
            idx_test = indices[int(y_len * train_splits) + int(y_len * val_splits):]

            self.train_mask = self.sample_mask(idx_train, y_len)
            self.val_mask = self.sample_mask(idx_val, y_len)
            self.test_mask = self.sample_mask(idx_test, y_len)

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

        self.num_features = data.num_features
        self.num_labels = y.max().item() + 1
        self.data_name = data_name

    def sample_mask(self, idx, l):
        """ create mask """
        mask = np.zeros(l)
        mask[idx] = 1
        return np.array(mask, dtype=np.int32)

if __name__=="__main__":

    graph = DATA()
    graph.get_data("cora",
                   train_splits=0.8,
                   val_splits=0.1,
                   shuffle_flag=True)

    pass

