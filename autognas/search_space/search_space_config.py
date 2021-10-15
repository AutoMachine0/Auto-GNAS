import os

class SearchSpace(object):
    """
    Loading the search space dict, config the number of stack gcn layer

    Args:
        gnn_layers:int
            config the number of sampling stack gcn layer

    Returns:
        component_value_dict:dict
            the search space component dict, the key is the stack gcn
            component and the value is the corresponding value
    """

    def __init__(self, gnn_layers=2):

        self.stack_gcn_architecture = ['attention',
                                       'aggregation',
                                       'multi_heads',
                                       'hidden_dimension',
                                       'activation'] * int(gnn_layers)

    def space_getter(self):

        search_space_path = os.path.split(os.path.realpath(__file__))[0]
        search_space_file_name = os.listdir(search_space_path)
        search_space_list = []

        for name in search_space_file_name:
            if ".py" in name:
                continue
            elif "__" in name:
                continue

            search_space_list.append(name)

        component_value_dict = {}

        for component in search_space_list:
            component_path = search_space_path + "/" + component
            value_name = os.listdir(component_path)
            value_list = []

            for name in value_name:
                if "__" in name:
                    continue
                value_list.append(name[:-3])

            component_value_dict[component] = value_list

        return component_value_dict


if __name__ == '__main__':

    stack_gnn_architecture = SearchSpace().stack_gcn_architecture
    component_value_dict = SearchSpace().space_getter()
    print(stack_gnn_architecture)
    print(component_value_dict)