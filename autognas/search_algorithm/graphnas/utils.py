import numpy as np
import torch
from torch.autograd import Variable

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]
    return x.item()

def experiment_time_save(path, file_name, train_gnn_sacle, gnn_scale, total_search_time):
    
    with open(path + "/" + file_name, "w") as f:

        f.write("controller training scale:" + str(train_gnn_sacle) + "\n" +
                "controller sample model scale:" + str(gnn_scale) + "\n" +
                "total search_algorithm time:" + str(total_search_time) + "s" + "\n")

    print("search cost time record done !")

def experiment_data_save(path, file_name, gnn_architecture_list, performance_list):
    
    performance_list_temp = performance_list.copy()
    best_performance = np.max(performance_list_temp)
    performance_list_temp.sort(reverse=True)

    if len(performance_list) > 10:
        top_avg_performance = np.mean(performance_list_temp[:10])
    else:
        top_avg_performance = np.mean(performance_list_temp[:len(performance_list)])
    
    with open(path + "/" + file_name, "w") as f:

        if len(performance_list) > 10:
            f.write("the best performance:\t" + str(best_performance) + "\t" +
                    "the top 10 avg performance:\t" + str(top_avg_performance) + "\n\n")
        else:
            f.write("the best performance:\t" + str(best_performance) + "\t" +
                    "the avg performance:\t" + str(top_avg_performance) + "\n\n")
        for gnn_architecture,  val_performance, in zip(gnn_architecture_list, performance_list):
            f.write(str(gnn_architecture)+";"+str(val_performance)+"\n")

    print("search process record done !")


