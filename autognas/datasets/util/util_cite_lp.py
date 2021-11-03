import os
import torch
import random
import numpy as np
from torch_geometric.datasets import Planetoid

class DATA(object):

    def __init__(self):

        self.name = ["cora_lp", "citeseer_lp", "pubmed_lp"]

    def get_data(self,
                 dataset,
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=123):

        if dataset == "cora_lp":
            dataset = "cora"
        elif dataset == "citeseer_lp":
            dataset = "citeseer"
        elif dataset == "pubmed_lp":
            dataset = "pubmed"

        data_name = dataset
        path = os.path.split(os.path.realpath(__file__))[0][:-14] + "/datasets/CITE/" + dataset
        dataset = Planetoid(path, dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = dataset[0]

        edge_index_ = data.edge_index
        edge_index = data.edge_index
        x = data.x

        edge_index = edge_index.tolist()
        edge_index_len = len(edge_index[0])

        # construct all node number set
        node_number_set = set([index for index in range(x.size(0))])

        # construct node mapping table
        mapping_dict = {}
        for index in range(edge_index_len):

            source_node_number = edge_index[0][index]
            target_node_number = edge_index[1][index]

            if source_node_number not in mapping_dict.keys():
                mapping_dict[source_node_number] = set()
                mapping_dict[source_node_number].add(target_node_number)
            else:
                if target_node_number not in mapping_dict[source_node_number]:
                    mapping_dict[source_node_number].add(target_node_number)
                else:
                    continue

        # dataset split
        indices = np.arange(edge_index_len).astype('int32')
        idx_train = indices[:int(edge_index_len * train_splits)]
        idx_val = indices[int(edge_index_len * train_splits):int(edge_index_len * train_splits) + int(edge_index_len * val_splits)]
        idx_test = indices[int(edge_index_len * train_splits) + int(edge_index_len * val_splits):]

        train_x_lp_source_node = np.array(edge_index[0])[idx_train]
        train_x_lp_target_node = np.array(edge_index[1])[idx_train]

        val_x_lp_source_node = np.array(edge_index[0])[idx_val]
        val_x_lp_target_node = np.array(edge_index[1])[idx_val]

        test_x_lp_source_node = np.array(edge_index[0])[idx_test]
        test_x_lp_target_node = np.array(edge_index[1])[idx_test]

        # construct negative link prediction for source node
        train_negative_source_list, \
        train_negative_target_list = self.negative_edge_getter(train_x_lp_source_node,
                                                               node_number_set,
                                                               mapping_dict)

        val_negative_source_list, \
        val_negative_target_list = self.negative_edge_getter(val_x_lp_source_node,
                                                             node_number_set,
                                                             mapping_dict)

        test_negative_source_list, \
        test_negative_target_list = self.negative_edge_getter(test_x_lp_source_node,
                                                              node_number_set,
                                                              mapping_dict)

        pos_train_edge_index = [list(train_x_lp_source_node),
                                list(train_x_lp_target_node)]

        neg_train_edge_index = [train_negative_source_list,
                                train_negative_target_list]

        pos_val_edge_index = [list(val_x_lp_source_node),
                              list(val_x_lp_target_node)]

        neg_val_edge_index = [val_negative_source_list,
                              val_negative_target_list]

        pos_test_edge_index = [list(test_x_lp_source_node),
                               list(test_x_lp_target_node)]

        neg_test_edge_index = [test_negative_source_list,
                               test_negative_target_list]

        # construct labels
        train_labels = self.get_link_labels(pos_train_edge_index,
                                            neg_train_edge_index)

        val_labels = self.get_link_labels(pos_val_edge_index,
                                          neg_val_edge_index)

        test_labels = self.get_link_labels(pos_test_edge_index,
                                           neg_test_edge_index)

        # construct training negative edge index list for training
        neg_train_edge_index_list = []

        for i in range(100):
            train_negative_source_list, \
            train_negative_target_list = self.negative_edge_getter(train_x_lp_source_node,
                                                                   node_number_set,
                                                                   mapping_dict)

            neg_train_edge_index = torch.tensor([train_negative_source_list,
                                                 train_negative_target_list],
                                                 dtype=torch.int64).to(device)

            neg_train_edge_index_list.append(neg_train_edge_index)

        self.train_x = [x.to(device)]
        self.val_x = [x.to(device)]
        self.test_x = [x.to(device)]

        self.train_edge_index = [edge_index_.to(device)]
        self.val_edge_index = [edge_index_.to(device)]
        self.test_edge_index = [edge_index_.to(device)]

        self.train_y = [torch.tensor(train_labels, dtype=torch.float32).to(device)]
        self.val_y = [torch.tensor(val_labels, dtype=torch.float32).to(device)]
        self.test_y = [torch.tensor(test_labels, dtype=torch.float32).to(device)]

        self.train_pos_edge_index = torch.tensor(pos_train_edge_index,
                                                 dtype=torch.int64).to(device)

        self.train_neg_edge_index_list = neg_train_edge_index_list

        self.val_pos_edge_index = torch.tensor(pos_val_edge_index,
                                               dtype=torch.int64).to(device)

        self.val_neg_edge_index = torch.tensor(neg_val_edge_index,
                                               dtype=torch.int64).to(device)

        self.test_pos_edge_index = torch.tensor(pos_test_edge_index,
                                                dtype=torch.int64).to(device)

        self.test_neg_edge_index = torch.tensor(neg_test_edge_index,
                                                dtype=torch.int64).to(device)

        self.num_features = data.num_features
        self.num_labels = 2
        self.data_name = data_name

    def negative_edge_getter(self,
                             source_node,
                             node_number_set,
                             mapping_dict):

        negative_source_list = []
        negative_target_list = []

        for index in range(len(source_node)):
            source_node_number = source_node[index]
            source_node_number_mapping_set = mapping_dict[source_node_number]
            sample_candidate_node = node_number_set - source_node_number_mapping_set
            negative_target_node_number = random.choice(list(sample_candidate_node))
            negative_source_list.append(source_node_number)
            negative_target_list.append(negative_target_node_number)

        return negative_source_list, negative_target_list

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = len(pos_edge_index[0]) + len(neg_edge_index[0])
        link_labels = np.array([0 for i in range(num_links)])
        link_labels[:len(pos_edge_index[0])] = 1.
        return link_labels

if __name__=="__main__":

    graph = DATA()
    graph.get_data("cora_lp",
                   train_splits=0.8,
                   val_splits=0.1,
                   shuffle_flag=True)