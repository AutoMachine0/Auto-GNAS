import torch
from numpy import *
from copy import deepcopy

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

class Batch(object):
    """
    Realizing the graph data mini batch
    Args:
        x: list
            the graph node embedding matrix list
            every element in the list is one graph node embedding matrix [N*d]
            N represents the number of node, d represent the the one node feature dimension
            the node embedding matrix type is tensor dim is [N*d]

        edge_index: list
            the graph edge index list
            every element in the list is one graph edge index matrix
            the edge index matrix type is tensor dim is [2*M]
            M represents the number of edge in one graph

        y: list
            the graph or node label matrix list
            every element in the list is one graph or node label matrix
            for graph label matrix dim is 1*1
            for node label matrix dim is 1*N
            N represents the number of node
            the element type is tensor

        batch_size: int
            mini batch size of graph data
    Returns:
        batch_x_list: list
            the graph node embedding matrix mini batch list
            every element in the list is one mini batch combine graph node embedding matrix
            the element dim is [(mini_batch_size*N)*d], if every graph node number is N
            the element type is tensor

        batch_edge_index_list: list
            the graph edge index matrix mini batch list
            every element in the list is one mini batch combine graph edge index matrix
            the element dim is [2*(M*mini_batch_size)]
            the element type is tensor

        batch_y_list: list
            the graph or node label mini batch matrix list
            for graph label mini batch matrix dim is [1*batch]
            for node label mini matrix dim is [1*N]
            N represents the number of node
            the element type is tensor

        batch_x_index_list: list
            the mini batch one graph node index list
            every element in the list is a tensor list contain the node number corresponding one graph
            the same number in the tensor list represents the corresponding node feature belong to the same graph
    """
    def __init__(self,
                 x: list,
                 edge_index: list,
                 y: list,
                 batch_size: int = 1):

        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = self.get()

    def get(self):

        x_len = len(self.x)

        if self.batch_size > x_len:

            print("the input data x length:", x_len)
            raise Exception("the batch size is out of input data x length, input batch_size:", self.batch_size)

        else:

            batch_x_list = []
            batch_edge_index_list = []
            batch_y_list = []
            batch_x_index_list = []

            batch_group = int(x_len / self.batch_size)
            index_start = 0
            index_end = self.batch_size

            for i in range(batch_group):

                batch_y = 0
                batch_x = 0
                batch_edge_index = 0
                batch_x_index = []
                index = 0

                first_x = True
                for x_, edge_index_, y_ in zip(self.x[index_start:index_end],
                                               self.edge_index[index_start:index_end],
                                               self.y[index_start:index_end]):
                    if first_x:
                        batch_x = x_
                        batch_edge_index = edge_index_
                        batch_y = y_
                        batch_x_index = [index for node in range(x_.shape[0])]
                        index += 1
                        first_x = False

                    else:
                        node_num = len(batch_x)
                        batch_x = torch.cat((batch_x, x_), 0)
                        batch_edge_index = torch.cat((batch_edge_index,
                                                      torch.add(edge_index_, node_num)), 1)
                        batch_y = torch.cat((batch_y, y_), 0)
                        batch_x_index = batch_x_index + [index for node in range(x_.shape[0])]
                        index += 1

                temp_batch_x = deepcopy(batch_x)
                temp_batch_edge_index = deepcopy(batch_edge_index)
                temp_batch_y = deepcopy(batch_y)
                temp_batch_x_index = deepcopy(batch_x_index)

                batch_x_list.append(temp_batch_x)
                batch_edge_index_list.append(temp_batch_edge_index)
                batch_y_list.append(temp_batch_y)
                batch_x_index_list.append(torch.tensor(temp_batch_x_index, dtype=torch.int64).to(self.device))

                index_start += self.batch_size
                index_end += self.batch_size

            # last batch process
            if (x_len % self.batch_size):

                batch_y = 0
                batch_x = 0
                batch_edge_index = 0
                batch_x_index = []
                index = 0

                first_x = True
                for x_, edge_index_, y_ in zip(self.x[index_start:],
                                               self.edge_index[index_start:],
                                               self.y[index_start:]):
                    if first_x:
                        batch_x = x_
                        batch_edge_index = edge_index_
                        batch_y = y_
                        batch_x_index = [index for node in range(x_.shape[0])]
                        index += 1
                        first_x = False
                    else:
                        node_num = len(batch_x)
                        batch_x = torch.cat((batch_x, x_), 0)
                        batch_edge_index = torch.cat((batch_edge_index,
                                                      torch.add(edge_index_, node_num)), 1)
                        batch_y = torch.cat((batch_y, y_), 0)
                        batch_x_index = batch_x_index + [index for node in range(x_.shape[0])]
                        index += 1

                batch_x_list.append(batch_x)
                batch_edge_index_list.append(batch_edge_index)
                batch_y_list.append(batch_y)
                batch_x_index_list.append(torch.tensor(batch_x_index, dtype=torch.int64).to(self.device))

        return batch_x_list, batch_edge_index_list, batch_y_list, batch_x_index_list