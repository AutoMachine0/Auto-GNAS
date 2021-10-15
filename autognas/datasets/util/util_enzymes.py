import os
import copy
import numpy as np
import torch
from random import shuffle

class DATA(object):

    def __init__(self):

        self.name = ["ENZYMES"]

    def get_data(self,
                 dataset="ENZYMES",
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=True):

        path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/ENZYMES/"

        file_name = ["ENZYMES_A.txt",
                     "ENZYMES_graph_indicator.txt",
                     "ENZYMES_graph_labels.txt",
                     "ENZYMES_node_attributes.txt",
                     "ENZYMES_node_labels.txt"]

        # obtain every graph node id
        node_id = 1
        graph_id_temp = 1
        graph_node_id_list = []
        sample_graph_node_id_list = []

        with open(path+file_name[1], 'r') as f:

            for graph_id in f.readlines():

                graph_id = int(graph_id.replace("\n", ""))

                if graph_id == graph_id_temp:
                    sample_graph_node_id_list.append(node_id)
                    node_id += 1
                else:
                    graph_node_id_list.append(copy.deepcopy(sample_graph_node_id_list))
                    graph_id_temp = graph_id
                    sample_graph_node_id_list = []
                    sample_graph_node_id_list.append(node_id)
                    node_id += 1

            graph_node_id_list.append(sample_graph_node_id_list)

        # initialize related data structure
        graph_edge_list = []
        graph_edge_list_temp = []
        node_features_list = []

        for i in range(len(graph_node_id_list)):

            graph_edge_list.append([[], []])
            graph_edge_list_temp.append([[], []])
            node_features_list.append([])

        # get every graph start node id for construct every graph edge list
        comparison_list = []

        for graph in graph_node_id_list:

            comparison_list.append(graph[0])

        # construct every graph edge list
        with open(path+file_name[0], 'r') as f:

            for edge in f.readlines():

                edge = edge.replace("\n", "").replace(" ", "").split(",")
                edge = [int(i) for i in edge]
                node_id = edge[0]

                for graph_start_node_id in comparison_list:

                    if node_id > graph_start_node_id:
                        if node_id > comparison_list[-1]:
                            index = comparison_list.index(comparison_list[-1])
                            graph_edge_list[index][0].append(edge[0])
                            graph_edge_list[index][1].append(edge[1])
                            break
                        else:
                            continue
                    elif node_id == graph_start_node_id:
                        index = comparison_list.index(graph_start_node_id)
                        graph_edge_list[index][0].append(edge[0])
                        graph_edge_list[index][1].append(edge[1])
                        break
                    elif node_id < graph_start_node_id:
                        index = comparison_list.index(graph_start_node_id) - 1
                        graph_edge_list[index][0].append(edge[0])
                        graph_edge_list[index][1].append(edge[1])
                        break

        # construct every graph edge list node id start with 0
        for index in range(len(graph_edge_list)):

            graph_start_node_id = comparison_list[index]
            graph_edge_list_temp[index][0] = [i - graph_start_node_id for i in graph_edge_list[index][0]]
            graph_edge_list_temp[index][1] = [i - graph_start_node_id for i in graph_edge_list[index][1]]

        # construct every graph node feature matrix
        node_features_temp = []

        with open(path+file_name[3], 'r') as f:

            for node_feature in f.readlines():

                node_feature = node_feature.replace("\n", "").replace(" ", "").split(",")
                node_feature = [float(i) for i in node_feature]
                node_features_temp.append(node_feature)

        for index in range(len(graph_node_id_list)):

            for node_index in graph_node_id_list[index]:

                node_features_list[index].append(node_features_temp[node_index-1])

        # get every graph label
        graph_label_list = []
        with open(path+file_name[2], 'r') as f:
            for graph_label in f.readlines():
                graph_label_list.append(int(graph_label.replace("\n", "")))

        # data type transformer for Auto-GNAS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        node_features_matrix_list = []
        for node_features_matrix in node_features_list:
            node_features_matrix_list.append(torch.tensor(np.array(node_features_matrix), dtype=torch.float32).to(device))

        graph_edge_list = []
        for graph_edge in graph_edge_list_temp:
            graph_edge_list.append(torch.tensor(np.array(graph_edge), dtype=torch.int64).to(device))

        # Auto-GNAS input required attribution
        y = []
        for graph_label in graph_label_list:
            y.append(torch.tensor([graph_label], dtype=torch.int64).to(device))

        if shuffle_flag:

            index_list = [i for i in range(len(y))]
            shuffle(index_list)
            shuffle_node_features_matrix_list = []
            shuffle_graph_edge_list = []
            shuffle_y = []

            for index in index_list:
                shuffle_node_features_matrix_list.append(node_features_matrix_list[index])
                shuffle_graph_edge_list.append(graph_edge_list[index])
                shuffle_y.append(y[index])

            node_features_matrix_list = shuffle_node_features_matrix_list
            graph_edge_list = shuffle_graph_edge_list
            y = shuffle_y

        if train_splits == None or val_splits == None:
            train_splits = 0.9
            val_splits = 0.05

        train_splits_end = int(len(node_features_matrix_list)*train_splits)
        val_splits_end = train_splits_end + int(len(node_features_matrix_list)*val_splits)

        self.train_x = node_features_matrix_list[:train_splits_end]
        self.val_x = node_features_matrix_list[train_splits_end:val_splits_end]
        self.test_x = node_features_matrix_list[val_splits_end:]

        self.train_y = y[:train_splits_end]
        self.val_y = y[train_splits_end:val_splits_end]
        self.test_y = y[val_splits_end:]

        self.train_edge_index = graph_edge_list[:train_splits_end]
        self.val_edge_index = graph_edge_list[train_splits_end:val_splits_end]
        self.test_edge_index = graph_edge_list[val_splits_end:]

        self.num_features = self.train_x[0].shape[1]
        self.num_labels = max(graph_label_list) + 1
        self.data_name = dataset

if __name__=="__main__":

    graph = DATA()
    graph.get_data('ENZYMES', shuffle_flag=True)
    pass