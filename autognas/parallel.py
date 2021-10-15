import ray
import math
import torch
from multiprocessing import cpu_count
from autognas.estimation import Estimation
from autognas.datasets.planetoid import Planetoid # for unit test

class ParallelConfig():
    """
    Controlling whether start the parallel estimation mode

    Args:
        none

    Returns:
        none
    """

    def __init__(self, ray_flag=True):

        ray.shutdown()
        if ray_flag:
            ray.init()  # parallel mode
        else:
            ray.init(local_mode=True)  # serial mode

class ParallelOperater(object):
    """
    Realizing the gnn architectures distribute and parallel estimation automatically
    1. recognizing machine GPU or CPU resources automatically
    2. dividing gnn architecture into different groups automatically
    3. parallel estimation model performance

    Args:
        data: graph data obj
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        gnn_parameter: dict
            the gnn model training validation testing config dict

    Returns:
        parallel_operator_list: list
            parallel estimator object list including multiple estimators
            that can estimate model performance at the same time
    """

    def __init__(self,
                 data,
                 gnn_parameter):

        self.data = data
        self.gnn_parameter = gnn_parameter
        self.parallel_operator_list = self.parallel_operator_initialize()

    def parallel_operator_initialize(self):
        """
        initialize parallel operator class
        """

        gpu_is, gpu_num, _, _ = self.gpu_check()
        cpu_logic_core_num = cpu_count()
        parallel_operator_list = []

        if gpu_is:
            print(35 * "=", "start using GPU estimation", 35 * "=")

            for parallel_operator_num in range(gpu_num):
                parallel_operator = GpuEstimation.remote(self.data,
                                                         self.gnn_parameter)
                parallel_operator_list.append(parallel_operator)
        else:
            print(35 * "=", "start using CPU estimation", 35 * "=")

            for parallel_operator_num in range(cpu_logic_core_num):
                parallel_operator = CpuEstimation.remote(self.data,
                                                         self.gnn_parameter)
                parallel_operator_list.append(parallel_operator)

        return parallel_operator_list

    def estimation(self, gnn_architecture_list):
        """
        Dividing the gnn architectures into different group,
        parallel estimating the model performance

        Args:
            gnn_architecture_list: list
                the stack gnn architecture describe
                for example,including one element stack gcn architecture list
                [['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']]

        Returns:
            result: list
                the performance list ,the element type is the float
        """

        group_num = len(self.parallel_operator_list)
        # model index group divide
        gnn_index_group_list = self.gnn_index_divide(group_num,
                                                     gnn_architecture_list)
        # model distribution based on group
        result = self.parallel_estimation(gnn_architecture_list,
                                          gnn_index_group_list,
                                          self.parallel_operator_list)

        return result

    def gnn_index_divide(self,
                         group_num,
                         gnn_architecture_list):

        # construct model index dividing manner based on group_num

        gnn_index_group_list = []
        if len(gnn_architecture_list) == 0 or not isinstance(gnn_architecture_list, list):
            raise Exception("wrong gnn_architecture_list:", gnn_architecture_list)

        if group_num < len(gnn_architecture_list):
            partition_num = math.ceil(len(gnn_architecture_list) / group_num)
            index = 0
            for i in range(partition_num):
                if (index + group_num - 1) < len(gnn_architecture_list):
                    gnn_index_group_list.append([index, index + group_num])
                    index += group_num
                else:
                    gnn_index_group_list.append([index, len(gnn_architecture_list)])
        elif group_num == len(gnn_architecture_list):
            index = 0
            for i in range(group_num):
                gnn_index_group_list.append([index, index + 1])
                index += 1
        elif group_num > len(gnn_architecture_list):
            index = 0
            for i in range(len(gnn_architecture_list)):
                gnn_index_group_list.append([index, index + 1])
                index += 1

        return gnn_index_group_list

    def parallel_estimation(self,
                            gnn_architecture_list,
                            gnn_index_group_list,
                            parallel_operator_list):

        result = []
        parallel_operator_num = len(parallel_operator_list)

        if parallel_operator_num < len(gnn_architecture_list):
            for gnn_group in gnn_index_group_list:
                task = []
                for gnn_architecture, parallel_operator in zip(gnn_architecture_list[gnn_group[0]:gnn_group[1]],
                                                               parallel_operator_list):
                    temp_performance = parallel_operator.estimation.remote(gnn_architecture)
                    task.append(temp_performance)
                for performacne in ray.get(task):
                    result.append(performacne)
                # release gpu memory
                torch.cuda.empty_cache()

        else:
            task = []
            for gnn_architecture, parallel_operator in zip(gnn_architecture_list, parallel_operator_list):
                temp_performance = parallel_operator.estimation.remote(gnn_architecture)
                task.append(temp_performance)
            result = [performance for performance in ray.get(task)]

            # release gpu memory
            torch.cuda.empty_cache()
        return result

    def gpu_check(self):

        gpu_is = torch.cuda.is_available()
        if gpu_is:
            gpu_num = torch.cuda.device_count()       # gpu number
            gpu_name = torch.cuda.get_device_name(0)  # gpu name
            gpu_id = torch.cuda.current_device()      # current server gpu name list
        else:
            gpu_num = 0
            gpu_name = None
            gpu_id = None

        return gpu_is, gpu_num, gpu_name, gpu_id

@ray.remote(num_gpus=1)
class GpuEstimation(object):

    def __init__(self, data, gnn_parameter):
        self.data = data
        self.gnn_parameter = gnn_parameter

    @ray.method(num_returns=1)
    def estimation(self, gnn_architecture):
        estimator = Estimation(gnn_architecture=gnn_architecture,
                               data=self.data,
                               gnn_parameter=self.gnn_parameter)
        performance = estimator.get_performance()
        print("gnn_architecture: " + str(gnn_architecture))
        print("performance: " + str(performance)+"\n")
        return performance

@ray.remote
class CpuEstimation(object):

    def __init__(self, data, gnn_parameter):
        self.data = data
        self.gnn_parameter = gnn_parameter

    @ray.method(num_returns=1)
    def estimation(self, gnn_architecture):
        estimator = Estimation(gnn_architecture=gnn_architecture,
                               data=self.data,
                               gnn_parameter=self.gnn_parameter)
        performance = estimator.get_performance()
        print("gnn_architecture: " + str(gnn_architecture))
        print("gnn_val_acc: " + str(performance) +"\n")
        return performance

if __name__=="__main__":

    graph = Planetoid("cora").data
    ParallelConfig(False)

    # test case 1: model num > parallel operator num
    gnn_list1 = [['generalized_linear', 'sum', 2, 32, 'linear', 'generalized_linear', 'sum', 2, 64, 'relu6'],
                ['const', 'sum', 1, 128, 'elu',  'linear', 'sum', 6, 8, 'sigmoid'],
                ['linear', 'sum', 2, 8, 'linear', 'generalized_linear', 'sum', 2, 32, 'elu'],
                ['linear', 'sum', 2, 32, 'elu', 'gat_sym', 'sum', 2, 128, 'softplus']]

    # test case 2: model num = parallel operator num
    gnn_list2 = [['generalized_linear', 'sum', 2, 32, 'linear',  'generalized_linear', 'sum', 2, 64, 'relu6'],
                ['const', 'sum', 1, 128, 'elu',  'linear', 'sum', 6, 8, 'sigmoid']]

    # test case 3: model num < parallel operator num
    gnn_list3 = [['generalized_linear', 'sum', 2, 32, 'linear', 'generalized_linear', 'sum', 2, 64, 'relu6']]

    gnn_parameter = {"gnn_type": "stack_gcn"}
    ParallelOperaterInstance = ParallelOperater(graph, gnn_parameter)
    result = ParallelOperaterInstance.estimation(gnn_list1)
    print("result:\n", result)