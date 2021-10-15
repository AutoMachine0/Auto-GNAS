from numpy import *

def data_information(data):

    the_graph_num = len(data.train_x)\
                    + len(data.val_x) \
                    + len(data.test_x)

    graph_size_list = []
    for graph in data.train_x:
        graph_size_list.append(graph.shape[0])
    for graph in data.val_x:
        graph_size_list.append(graph.shape[0])
    for graph in data.test_x:
        graph_size_list.append(graph.shape[0])

    graph_max_size = max(graph_size_list)
    graph_min_size = min(graph_size_list)
    graph_mean_size = mean(graph_size_list)

    edge_size_list = []
    for graph_edge_index in data.train_edge_index:
        edge_size_list.append(graph_edge_index[0].shape[0])
    for graph_edge_index in data.val_edge_index:
        edge_size_list.append(graph_edge_index[0].shape[0])
    for graph_edge_index in data.test_edge_index:
        edge_size_list.append(graph_edge_index[0].shape[0])

    graph_edge_max_size = max(edge_size_list)
    graph_edge_min_size = min(edge_size_list)
    graph_edge_mean_size = mean(edge_size_list)

    node_feature_num = data.num_features
    label_num = data.num_labels
    data_name = data.data_name

    data_information_dict = {"data_name": data_name,
                             "the graph num": the_graph_num,
                             "graph_max_size": graph_max_size,
                             "graph_min_size": graph_min_size,
                             "graph_mean_size": graph_mean_size,
                             "graph_edge_max_size": graph_edge_max_size,
                             "graph_edge_min_size": graph_edge_min_size,
                             "graph_edge_mean_size": graph_edge_mean_size,
                             "node_feature_num": node_feature_num,
                             "label_num": label_num}

    return data_information_dict