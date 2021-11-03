from torch_geometric.datasets import Entities
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset

path = os.path.split(os.path.realpath(__file__))[0][:-21] + "/autognas/datasets/MUTAG_/"

mutag = Entities(path, "MUTAG", T.NormalizeFeatures())
#mutag = dataset = TUDataset(path, name="MUTAG")
pass