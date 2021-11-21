import os
import torch.nn
import importlib

def search_algorithm_getter(search_algorithm_type,
                            data,
                            search_parameter,
                            gnn_parameter,
                            search_space):
    """
    Dynamically obtain search algorithm obj

    Args:
        search_algorithm_type: str
            the search algorithm type
        data: graph data object
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        search_parameter: dict
            the search parameter in the search process
        search_space: search space object
            the search space object including search space
            dict and gnn architecture list

    Returns:
        search_algorithm: search algorithm object
            realizing the gnn architecture sample
    """

    search_algorithm_class = "Search"
    search_algorithm_module = importlib.import_module("autognas.search_algorithm." +
                                                      search_algorithm_type +
                                                      "." +
                                                      "search_algorithm")

    search_algorithm_obj = getattr(search_algorithm_module,
                                   search_algorithm_class)

    search_algorithm = search_algorithm_obj(data,
                                            search_parameter,
                                            gnn_parameter,
                                            search_space)
    return search_algorithm

def optimizer_getter(opt_type,
                     gnn_model,
                     optimizer_parameter_dict):
    """
    Dynamically obtain optimizing function obj

    Args:
        opt_type: str
            the optimizer function type
        gnn_model: gnn model object
            the gnn model object which will be trained
        optimizer_parameter_dict: dict
            the hyper parameter of optimizer function

    Returns:
        optimizer_function: optimizer function object
            the optimizer function object for gnn model training
    """

    optimizer_class = "Optimizer"
    optimizer_module = importlib.import_module("autognas.model.optimizer_function" +
                                               "." +
                                               opt_type)
    optimizer_obj = getattr(optimizer_module, optimizer_class)
    optimizer_function = optimizer_obj(gnn_model,
                                       optimizer_parameter_dict).function()

    return optimizer_function


def loss_getter(loss_type):
    """
    Dynamically obtain loss function obj

    Args:
        loss_type: str
            the loss function type

    Returns:
        loss_function: loss function object
            the loss function object for gnn model training
    """

    loss_class = "Loss"
    loss_module = importlib.import_module("autognas.model.loss_function" +
                                          "." +
                                          loss_type)
    loss_obj = getattr(loss_module, loss_class)
    loss_function = loss_obj()

    return loss_function


def evaluator_getter(evaluator_type):
    """
    Dynamically obtain evaluator function obj

    Args:
        evaluator_type: str
            the evaluator function type

    Returns:
        evaluator_function: evaluator function obj
            the evaluator function object for gnn model validation and testing

    """

    evaluator_class = "Evaluator"
    evaluator_module = importlib.import_module("autognas.model.evaluator_function" +
                                               "." +
                                               evaluator_type)
    evaluator_obj = getattr(evaluator_module, evaluator_class)
    evaluator_function = evaluator_obj()

    return evaluator_function


def downstream_task_model_getter(downstream_task_type,
                                 gnn_embedding_dim,
                                 graph_data):
    """
    Dynamically obtain downstream task model obj

    Args:
        downstream_task_type: str
            the downstream task type
        gnn_embedding_dim: int
            the output dimension of gnn model
        graph_data: graph data object
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name

    Returns:
        downstream_task: downstream task model object
            realizing the downstream task forwarding computing process object
    """

    downstream_task_model_class = "DownstreamTask"
    downstream_task_model_module = importlib.import_module("autognas.model.downstream_task_model" +
                                                           "." +
                                                           downstream_task_type)
    downstream_task_model_obj = getattr(downstream_task_model_module, downstream_task_model_class)

    downstream_task = downstream_task_model_obj(gnn_embedding_dim, graph_data)

    return downstream_task

def attention_getter(heads,
                     output_dim):
    """
    Dynamically obtain attention function obj from search_algorithm space

    Args:
        heads: int
            the number of multi heads
        output_dim: int
            the output dimension in this gnn layer

    Returns:
        attention_dict: dict
            the attention function object dict, the key is the type of attention function,
            the value is the corresponding attention function object
    """

    search_space_path = os.path.split(os.path.realpath(__file__))[0] + "/search_space/attention"
    attention_list = [attention for attention in os.listdir(search_space_path) if attention not in "__pycache__"
                      and attention not in "README.md"]
    attention_dict = torch.nn.ModuleDict()
    for attention in attention_list:
        attention_class = "Attention"
        attention_module = importlib.import_module("autognas.search_space.attention" +
                                                    "." + attention[:-3])
        attention_obj = getattr(attention_module, attention_class)
        attention_function = attention_obj(heads, output_dim)
        attention_dict[attention[:-3]] = attention_function

    return attention_dict

def aggregation_getter():
    """
    Dynamically obtain aggregation function obj from search_algorithm space

    Args:
        none

    Returns:
        aggregation_dict: dict
            the aggregation function object dict, the key is the type of aggregation function,
            the value is the corresponding aggregation function object
            the max pooling, mean pooling, sum aggregation manner are realized by PYG
    """

    search_space_path = os.path.split(os.path.realpath(__file__))[0]+ "/search_space/aggregation"
    aggregation_list = [aggregation for aggregation in os.listdir(search_space_path) if aggregation not in "__pycache__"
                        and aggregation not in "README.md"]
    aggregation_dict = {}
    for aggregation in aggregation_list:
        aggregation_class = "Aggregation"
        aggregation_module = importlib.import_module("autognas.search_space.aggregation" +
                                                     "." + aggregation[:-3])
        aggregation_obj = getattr(aggregation_module, aggregation_class)
        aggregation_function = aggregation_obj()
        aggregation_dict[aggregation[:-3]] = aggregation_function

    return aggregation_dict

def activation_getter():
    """
    Dynamically obtain activation function obj from search_algorithm space

    Args:
        none

    Returns:
        activation_dict: dict
            the activation function object dict, the key is the type of activation function,
            the value is the corresponding activation function object
    """

    search_space_path = os.path.split(os.path.realpath(__file__))[0] + "/search_space/activation"
    activation_list = [activation for activation in os.listdir(search_space_path) if activation not in "__pycache__"
                       and activation not in "README.md"]
    activation_dict = {}
    for activation in activation_list:
        activation_class = "Activation"
        activation_module = importlib.import_module("autognas.search_space.activation" +
                                                    "." + activation[:-3])
        activation_obj = getattr(activation_module, activation_class)
        activation_function = activation_obj().function()
        activation_dict[activation[:-3]] = activation_function

    return activation_dict

def data_util_class_getter():
    """
        Dynamically obtain default dataset util object

        Args:
            none

        Returns:
            data_util_dict: dict
                the data util object dict, the key is the dataset name
                the value is the corresponding data util object
        """

    data_util_class_path = os.path.split(os.path.realpath(__file__))[0] + "/datasets/util"

    data_util_script_name_list = [data_util_class for data_util_class in os.listdir(data_util_class_path) if
                            data_util_class not in "__pycache__" and data_util_class not in "__init__.py"]
    data_util_dict = {}

    for data_util_script_name in data_util_script_name_list:

        class_name = "DATA"
        data_util_script = importlib.import_module("autognas.datasets.util" +
                                                   "." + data_util_script_name[:-3])
        data_util_obj = getattr(data_util_script, class_name)
        data_util_obj = data_util_obj()
        data_util_dict[data_util_obj] = data_util_obj.name

    return data_util_dict

if __name__=="__main__":
    # data_util_class_path = os.path.split(os.path.realpath(__file__))[0]
    # print(data_util_class_path)

    attention_dict = attention_getter(1, 64)
    pass