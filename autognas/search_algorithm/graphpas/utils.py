import numpy as np

def experiment_graphpas_data_save(path,
                                  file_name,
                                  gnn_architecture_list,
                                  performance_list,
                                  search_space,
                                  stack_gcn_architecture):

    performance_list_temp = performance_list.copy()
    best_performance = np.max(performance_list_temp)
    performance_list_temp.sort(reverse=True)

    if len(performance_list) > 10:
        top_avg_performance = np.mean(performance_list_temp[:10])
    else:
        top_avg_performance = np.mean(performance_list_temp[:len(performance_list)])

    gnn_architecture_list_temp = []

    with open(path + "/" + file_name, "w") as f:

        if len(performance_list) > 10:
            f.write("the best performance:\t" + str(best_performance) + "\t" +
                    "the top 10 avg performace:\t" + str(top_avg_performance) + "\n\n")
        else:
            f.write("the best performance:\t" + str(best_performance) + "\t" +
                    "the avg performace:\t" + str(top_avg_performance) + "\n\n")

        for gnn_architecture_embedding in gnn_architecture_list:
            gnn_architecture_list_temp.append(gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                                 search_space,
                                                                                 stack_gcn_architecture))
        for gnn_architecture,  val_performance, in zip(gnn_architecture_list_temp, performance_list):
            f.write(str(gnn_architecture)+";"+str(val_performance)+"\n")


    print("search process record done !")

def experiment_time_save(path,
                         file_name,
                         epoch,
                         time_cost):

    with open(path + "/" + file_name, "w") as f:
        for epoch_, timestamp, in zip(epoch, time_cost):
            f.write(str(epoch_)+";"+str(timestamp)+"\n")
    print("search time cost record done !")

def experiment_time_save_initial(path,
                                 file_name,
                                 time_cost):

    with open(path + "/" + file_name, "w") as f:
        f.write(str(time_cost)+"\n")
    print("initial time cost record done !")

def mutation_selection_probability(sharing_population,
                                   gnn_architecture_flow):

    p_list = []

    for i in range(len(gnn_architecture_flow)):

        p_list.append([])

    while sharing_population:

        gnn = sharing_population.pop()

        for index in range(len(p_list)):

            p_list[index].append(gnn[index])

    gene_information_entropy = []

    for sub_list in p_list:

        gene_information_entropy.append(information_entropy(sub_list))

    exp_x = np.exp(gene_information_entropy)
    probability = exp_x / np.sum(exp_x)

    return probability

def information_entropy(p_list):

    dict = {}
    length = len(p_list)

    for key in p_list:

        dict[key] = dict.get(key, 0) + 1

    p_list = []

    for key in dict:

        p_list.append(dict[key] / length)

    p_array = np.array(p_list)
    log_p = np.log2(p_array)
    information_entropy = -sum(p_array * log_p)

    return information_entropy

def top_population_select(population,
                          performanceuracy, top_k):

    population_dict = {}

    for key, value in zip(population, performanceuracy):

        population_dict[str(key)] = value
    rank_population_dict = sorted(population_dict.items(), key=lambda x: x[1], reverse=True)

    sharing_popuplation = []
    sharing_validation_performance = []

    i = 0
    for key, value in rank_population_dict:

        if i == top_k:
            break
        else:
            sharing_popuplation.append(eval(key))
            sharing_validation_performance.append(value)
            i += 1
    return sharing_popuplation, sharing_validation_performance

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