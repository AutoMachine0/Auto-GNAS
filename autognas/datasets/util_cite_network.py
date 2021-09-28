import os
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

class CiteNetwork():

    def __init__(self, dataset):
        if dataset in ["cora", "citeseer", "pubmed"]:
            data_name = dataset
            path = os.path.split(os.path.realpath(__file__))[0][:-9] + "/datasets/cite_network/" + dataset
            dataset = Planetoid(path, dataset, T.NormalizeFeatures())
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            data = dataset[0]

            self.edge_index = data.edge_index.to(device)
            self.x = data.x.to(device)
            self.y = data.y.to(device)

            self.train_mask = data.train_mask.to(device)
            self.val_mask = data.val_mask.to(device)
            self.test_mask = data.test_mask.to(device)

            self.train_y = self.y[self.train_mask]
            self.val_y = self.y[self.val_mask]
            self.test_y = self.y[self.test_mask]

            self.train_x = self.x
            self.val_x = self.x
            self.test_x = self.x

            self.train_edge_index = self.edge_index
            self.val_edge_index = self.edge_index
            self.test_edge_index = self.edge_index

            self.num_features = data.num_features
            self.data_name = data_name
        else:
            print("wrong graph data name")

if __name__=="__main__":

    data_name = "citeseer"
    graph = CiteNetwork(data_name)
    pass


