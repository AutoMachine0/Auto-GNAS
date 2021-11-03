import os
import time
from autognas.search_algorithm.random import utils
from autognas.parallel import ParallelOperater, \
                              ParallelConfig

from autognas.datasets.planetoid import Planetoid # for unit test
from autognas.search_space.search_space_config import SearchSpace  # for unit test


class RandomSearch(object):

    def __init__(self,
                 gnn_scale,
                 search_space):

        self.gnn_scale = gnn_scale
        self.gnn_architecture_list = []
        self.search_space = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture

    def search(self):

        print(35*"=", "random search", 35*"=")

        while len(self.gnn_architecture_list) < self.gnn_scale:

            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)
            self.gnn_architecture_list.append(gnn_architecture)

        return self.gnn_architecture_list

class Search(object):

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.search_space = search_space

    def search_operator(self):
        start_time = time.time()

        print(35 * "=", "random search start", 35 * "=")
        searcher = RandomSearch(int(self.search_parameter["gnn_scale"]),
                                self.search_space)

        # Parallel Operator Module Initialize
        parallel_estimation = ParallelOperater(self.data, self.gnn_parameter)

        gnn_architecture_list = searcher.search()
        result = parallel_estimation.estimation(gnn_architecture_list)

        search_total_time = time.time() - start_time

        # model architecture / val acc record
        path = os.path.split(os.path.realpath(__file__))[0][:-32] + "/logger/random_logger/"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_data_save(path, self.data.data_name + ".txt", gnn_architecture_list, result)

        index = result.index(max(result))
        best_val_architecture = gnn_architecture_list[index]
        best_acc = max(result)
        print("Best architecture:\n", best_val_architecture)
        print("Best val_performance:\n", best_acc)

        # search_algorithm total time record
        path = os.path.split(os.path.realpath(__file__))[0][:-32] + "/logger/random_logger/"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_total_time.txt",
                                   len(gnn_architecture_list),
                                   search_total_time)

if __name__=="__main__":

    # ParallelConfig(True)
    ParallelConfig(False)

    # get graph
    graph = Planetoid("cora").data

    # search_algorithm config
    search_parameter = {"gnn_scale": "10"}

    # GNNs config
    gnn_parameter = {"gnn_type": "stack_gcn",
                     "downstream_task_type": "node_classification",
                     "train_batch_size": "1",
                     "val_batch_size": "1",
                     "test_batch_size": "1",
                     "gnn_drop_out": "0.6",
                     "train_epoch": "10",
                     "early_stop": "False",
                     "early_stop_patience": "10",
                     "opt_type": "adam",
                     "opt_type_dict": "{\"learning_rate\": 0.005, \"l2_regularization_strength\": 0.0005}",
                     "loss_type": "nll_loss",
                     "val_evaluator_type": "accuracy",
                     "test_evaluator_type": "[\"accuracy\", \"precision\", \"recall\", \"f1_value\"]"}

    search_space = SearchSpace(gnn_layers="2")
    graphpas_instance = Search(graph, search_parameter, gnn_parameter,search_space)
    graphpas_instance.search_operator()