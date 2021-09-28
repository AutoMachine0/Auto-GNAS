import os
import configparser
from autognas.auto_model import AutoModel
from autognas.parallel import ParallelConfig
from autognas.datasets.util_cite_network import CiteNetwork

ParallelConfig(False)
graph = CiteNetwork("cora")
config = configparser.ConfigParser()

config_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + "/config/"

configuration = ["genetic.ini", "graphpas.ini", "graphnas.ini", "random.ini"]

for sub_config in configuration:
    config.read(config_path+sub_config)
    search_parameter = dict(config.items('search_parameter'))
    gnn_parameter = dict(config.items("gnn_parameter"))
    AutoModel(graph, search_parameter, gnn_parameter)