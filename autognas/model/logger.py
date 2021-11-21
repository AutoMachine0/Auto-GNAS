import os
import torch
import time

logger_path = os.path.split(os.path.realpath(__file__))[0][:-15] + "/logger"

def gnn_architecture_performance_save(gnn_architecture, performance, data_name):

    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    with open(logger_path + "/gnn_logger_" + str(data_name) + ".txt", "a+") as f:
        f.write(str(gnn_architecture) + ":" + str(performance) + "\n")

    print("gnn architecture and validation performance save")
    print("save path: ", logger_path + "/gnn_logger_" + str(data_name) + ".txt")
    print(50 * "=")

def test_performance_save(gnn_architecture, test_performance_dict, hyperparameter_dict, data_name):

    if not os.path.exists(logger_path):
        os.makedirs(logger_path)

    file_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    with open(logger_path + "/test_logger_" + data_name + "_" + file_time + ".txt", "a+") as f:
        f.write("gnn architecture:\t" + str(gnn_architecture)+ "\n")
        f.write(25 * "=" + " hyperparameter " + 25 * "=" + "\n")
        for hyperparameter in hyperparameter_dict.keys():
            f.write(str(hyperparameter) + ":" + str(hyperparameter_dict[hyperparameter])+"\n")
        f.write(25*"=" + " test performance result " + 25*"=" + "\n")
        for performance in test_performance_dict.keys():
            f.write(str(performance) + ":" + str(test_performance_dict[performance])+"\n")
        f.write(50 * "=" + "\n\n")

    print("hyperparameter and test performance save")
    print("save path: ", logger_path + "/test_logger_" + data_name + "_" + file_time + ".txt")

def model_save(gnn_model, optimizer, data_name, model_num):

    state = {"gnn_model": gnn_model.state_dict(),
             "optimizer": optimizer.state_dict()}
    torch.save(state, logger_path+"/model_" + data_name + "_" + str(model_num) + ".pth")

    print("gnn model and optimizer parameter save")
    print("save path: ", logger_path+"/model.pth")

if __name__=="__main__":
    gnn = ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 7, 'tanh']
    performance = 0.89
    gnn_architecture_performance_save(gnn, performance, "data_name")