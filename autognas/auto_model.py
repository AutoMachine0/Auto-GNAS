import os
from autognas.estimation import Estimation
from autognas.parallel import ParallelConfig
from autognas.search_space.search_space_config import SearchSpace
from autognas.dynamic_configuration import search_algorithm_getter

from autognas.datasets.util_cite_network import CiteNetwork  # for unit test
import configparser

class AutoModel(object):
    """
    The top API to realize gnn architecture search and model testing automatically.

    Using search algorithm samples gnn architectures and evaluate
    corresponding performance,testing the top k model from the sampled
    gnn architectures based on performance.

    Args:
        data: graph data obj
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.data_name
        search_parameter: dict
            the search algorithm configuration dict to control the
            automatic search process including required attributes:
            1.search_algorithm_type, 2.test_gnn_num
        gnn_parameter: dict
            the gnn configuration dict to complete the gnn model train
            validate and test based on the gnn architecture, for the
            stack gcn architecture, the required attributes includes:
            1.gnn_type,2.gnn_layers,3.downstream_task_type,
            4.gnn_drop_out,5.train_epoch,6.early_stop
            7.early_stop_patience,8.opt_type,9.opt_type_dict
            10.loss_type,11.val_evaluator_type,12.val_evaluator_type

    Returns:
        None
    """

    def __init__(self, data, search_parameter, gnn_parameter):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.search_space = SearchSpace(self.gnn_parameter["gnn_layers"])

        print("search parameter information:\t", self.search_parameter)
        print("gnn parameter information:\t", self.gnn_parameter)
        print("search space information:\t", self.search_space.space_getter())
        print("stack gcn architecture information:\t", self.search_space.stack_gcn_architecture)

        self.search_algorithm = search_algorithm_getter(self.search_parameter["search_algorithm_type"],
                                                        self.data,
                                                        self.search_parameter,
                                                        self.gnn_parameter,
                                                        self.search_space)


        self.search_model()

        self.derive_target_model()

    def search_model(self):

        self.search_algorithm.search_operator()

    def derive_target_model(self):

        path = os.path.split(os.path.realpath(__file__))[0][:-9] + "/logger/gnn_logger_"
        architecture_performance_list = self.gnn_architecture_performance_load(path, self.data.data_name)
        gnn_architecture_performance_dict = {}
        gnn_architecture_list = []
        performance_list = []

        for line in architecture_performance_list:
            line = line.split(":")
            gnn_architecture = eval(line[0])
            performance = eval(line[1].replace("\n", ""))
            gnn_architecture_list.append(gnn_architecture)
            performance_list.append(performance)

        for key, value in zip(gnn_architecture_list, performance_list):
            gnn_architecture_performance_dict[str(key)] = value

        ranked_gnn_architecture_performance_dict = sorted(gnn_architecture_performance_dict.items(),
                                                          key=lambda x: x[1],
                                                          reverse=True)

        sorted_gnn_architecture_list = []
        sorted_performance = []

        top_k = int(self.search_parameter["test_gnn_num"])
        i = 0
        for key, value in ranked_gnn_architecture_performance_dict:
            if i == top_k:
                break
            else:
                sorted_gnn_architecture_list.append(eval(key))
                sorted_performance.append(value)
                i += 1

        model_num = [num for num in range(len(sorted_gnn_architecture_list))]

        print(35*"=" + " the testing start " + 35*"=")

        for target_architecture, num in zip(sorted_gnn_architecture_list, model_num):

            testor = Estimation(target_architecture, self.data, self.gnn_parameter)
            testor.get_test_result(num)

        print(35 * "=" + " the testing ending " + 35 * "=")

    def gnn_architecture_performance_load(self, path, data_name):

        with open(path + data_name + ".txt", "r") as f:
            gnn_architecture_performance_list = f.readlines()
        return gnn_architecture_performance_list

if __name__ =="__main__":

    ParallelConfig(False)
    graph = CiteNetwork("cora")
    conf = configparser.ConfigParser()
    config_path = os.path.abspath(os.path.join(os.getcwd(), "../")) + "/config/"
    conf.read(config_path + "genetic.ini")
    search_parameter = dict(conf.items('search_parameter'))
    gnn_parameter = dict(conf.items("gnn_parameter"))
    graphpas_instance = AutoModel(graph, search_parameter, gnn_parameter)