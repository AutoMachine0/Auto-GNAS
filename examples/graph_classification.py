import os
import configparser
from autognas.auto_model import AutoModel
from autognas.parallel import ParallelConfig
from autognas.datasets.planetoid import Planetoid

# ParallelConfig(True)
ParallelConfig(False)

graph = Planetoid("PROTEINS",
                  shuffle_flag=True,
                  train_batch_size=100,
                  val_batch_size=10,
                  test_batch_size=10).data

config = configparser.ConfigParser()

config_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/config/graph_classification_config/"

configuration = ["graphpas.ini"]

for sub_config in configuration:
    config.read(config_path+sub_config)
    search_parameter = dict(config.items('search_parameter'))
    gnn_parameter = dict(config.items("gnn_parameter"))
    AutoModel(graph, search_parameter, gnn_parameter)