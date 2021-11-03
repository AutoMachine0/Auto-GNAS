import torch
import copy
from torch_scatter import scatter_mean, scatter_add, scatter_max

def batch_util(batch_size, x, edge_index, y):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_len = len(x)

    if batch_size > x_len:

        print("the input data x length:", x_len)
        raise Exception("the batch size is out of input data x length, input batch_size:", batch_size)

    else:

        batch_x_list = []
        batch_edge_index_list = []
        batch_y_list = []
        batch_x_index_list = []

        batch_group = int(x_len / batch_size)
        index_start = 0
        index_end = batch_size

        for i in range(batch_group):

            batch_y = 0
            batch_x = 0
            batch_edge_index = 0
            batch_x_index = []
            index = 0

            first_x = True
            for x_, edge_index_, y_ in zip(x[index_start:index_end],
                                           edge_index[index_start:index_end],
                                           y[index_start:index_end]):
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

            temp_batch_x = copy.deepcopy(batch_x)
            temp_batch_edge_index = copy.deepcopy(batch_edge_index)
            temp_batch_y = copy.deepcopy(batch_y)
            temp_batch_x_index = copy.deepcopy(batch_x_index)

            batch_x_list.append(temp_batch_x)
            batch_edge_index_list.append(temp_batch_edge_index)
            batch_y_list.append(temp_batch_y)
            batch_x_index_list.append(torch.tensor(temp_batch_x_index, dtype=torch.int64).to(device))

            index_start += batch_size
            index_end += batch_size

        # last batch process
        if (x_len % batch_size):

            batch_y = 0
            batch_x = 0
            batch_edge_index = 0
            batch_x_index = []
            index = 0

            first_x = True
            for x_, edge_index_, y_ in zip(x[index_start:],
                                           edge_index[index_start:],
                                           y[index_start:]):
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
            batch_x_index_list.append(torch.tensor(batch_x_index, dtype=torch.int64).to(device))

    return batch_x_list, batch_edge_index_list, batch_y_list, batch_x_index_list

def node_embedding_mean_pooling_to_graph_embedding(batch_node_embedding_matrix,
                                                   index):

    graph_embedding = scatter_mean(batch_node_embedding_matrix, index, dim=0)
    return graph_embedding

def node_embedding_sum_pooling_to_graph_embedding(batch_node_embedding_matrix,
                                                  index):

    graph_embedding = scatter_add(batch_node_embedding_matrix, index, dim=0)
    return graph_embedding

def node_embedding_max_pooling_to_graph_embedding(batch_node_embedding_matrix,
                                                  index):

    graph_embedding, _ = scatter_max(batch_node_embedding_matrix, index, dim=0)
    return graph_embedding