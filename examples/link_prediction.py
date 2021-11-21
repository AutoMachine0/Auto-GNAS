import os
import configparser
from autognas.auto_model import AutoModel
from autognas.parallel import ParallelConfig
from autognas.datasets.planetoid import Planetoid

ParallelConfig(True)
graph = Planetoid("cora_lp", train_splits=0.85, val_splits=0.05).data
config = configparser.ConfigParser()

config_path = os.path.abspath(os.path.join(os.getcwd(), "..")) + "/config/link_prediction_config/"

configuration = ["graphnas.ini", "graphpas.ini", "random.ini", "genetic.ini"]

for sub_config in configuration:
    config.read(config_path+sub_config)
    search_parameter = dict(config.items('search_parameter'))
    gnn_parameter = dict(config.items("gnn_parameter"))
    AutoModel(graph, search_parameter, gnn_parameter)