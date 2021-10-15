import numpy as np

# data read / wirte
def experiment_data_record(path,
                           file_name,
                           gnn_architecture_list,
                           performance_list):

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

def experiment_time_save(path, file_name, epoch, time_cost):

    total_time_cost = np.sum(time_cost)

    with open(path + "/" + file_name, "w") as f:

        f.write("the total search cost:\t" + str(total_time_cost) + "s" + "\n\n")

        for epoch_, timestamp, in zip(epoch, time_cost):

            f.write(str(epoch_) + ":" + str(timestamp) + "s" + "\n")

    print("search time cost record done !")

def experiment_time_save_initial(path, file_name, time_cost):

    with open(path + "/" + file_name, "w") as f:

        f.write(str(time_cost)+"\n")

    print("initial time cost record done !")

# select top population based on fitness
def top_population_select(population, performanceuracy, top_k):

    population_dict = {}
    for key, value in zip(population, performanceuracy):
        population_dict[str(key)] = value

    # rank based on fitness
    rank_population_dict = sorted(population_dict.items(), key=lambda x: x[1], reverse=True)

    top_popuplation = []
    top_fitness = []
    i = 0
    for key, value in rank_population_dict:

        if i == top_k:
            break
        else:
            top_popuplation.append(eval(key))
            top_fitness.append(value)
            i += 1
    return top_popuplation, top_fitness

def gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                       search_space,
                                       stack_gcn_architecture):
    gnn_architecture = []

    for component_embedding, component_name in zip(gnn_architecture_embedding, stack_gcn_architecture):

        component = search_space[component_name][component_embedding]
        gnn_architecture.append(component)

    return gnn_architecture

def random_generate_gnn_architecture_embedding(search_space,
                                               stack_gcn_architecture):
    gnn_architecture_embedding = []

    for component in stack_gcn_architecture:

        gnn_architecture_embedding.append(np.random.randint(0, len(search_space[component])))

    return gnn_architecture_embedding

if __name__=="__main__":
    pass