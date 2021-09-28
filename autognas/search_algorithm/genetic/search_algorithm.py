import numpy as np
import os
import time
from autognas.search_algorithm.genetic import utils
from autognas.parallel import ParallelOperater, \
                              ParallelConfig

from autognas.datasets.util_cite_network import CiteNetwork  # for unit test
from autognas.search_space.search_space_config import SearchSpace  # for unit test

class GeneticSearch(object):

    def __init__(self,
                 search_space,
                 initial_population_scale=100,
                 parent_scale=20,
                 crossover_rate=1.0,
                 mutation_rate=0.02,
                 update_threshold=20):

        if parent_scale > initial_population_scale:
            raise Exception("wrong parent scale should be smaller than initial_population_scale,"
                            "the current parent scale:", parent_scale)

        self.initial_population_scale = initial_population_scale
        self.parent_scale = parent_scale
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.update_threshold = update_threshold
        self.population_embedding = []
        self.population = []
        self.fitness = []
        self.search_space = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture

        self.random_initialize_population()

    def search(self):

        print(35*"=", "genetic search_algorithm", 35*"=")

        # select parent based on wheel strategy
        parents = self.selection(self.population_embedding, self.fitness, self.parent_scale)
        print(35*"=", "parents:", 35*"=", "\n", parents)

        # crossover based on crossover rate
        children = self.crossover(parents)

        # mutation based on mutarion rate
        children = self.mutation(children)

        print(35*"=", "new children:",  35*"=", "\n",  children,)

        return children

    def random_initialize_population(self):

        print(35*"=", "population initializing based on random strategy", 35*"=")

        while len(self.population) < self.initial_population_scale:

            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)

            self.population_embedding .append(gnn_architecture_embedding)
            self.population.append(gnn_architecture)

    def selection(self, population_embedding, fitness, parent_scale):

        print(35*"=", " wheel select ", 35*"=")

        # calculate selection probability based on fitness
        fitness = np.array(fitness)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()

        # select parents based on selection probability
        index_list = [index for index in range(len(fitness))]
        parents = []
        parent_index = np.random.choice(index_list, parent_scale, replace=False, p=fitness_probility)

        for index in parent_index:
            parents.append(population_embedding[index].copy())

        return parents

    def crossover(self, parents):

        print(35 * "=", "single point crossover", 35 * "=")
        print("before crossover:\n", parents)

        if (len(parents) % 2) != 0:
            raise Exception("wrong len(parents),len(parents) should be even number:", len(parents))

        children = []
        while parents:
            # step1: get a couple of parent without replace
            parent_1 = parents.pop()
            parent_2 = parents.pop()
            # step2: identify crossover point based on random
            crossover_point = np.random.randint(1, len(parent_1))
            # step3: crossover
            corssover_op = np.random.choice([True, False], 1, p=[self.crossover_rate, 1 - self.crossover_rate])[0]

            if corssover_op:
                child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
                child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
            else:
                child_1 = parent_1
                child_2 = parent_2

            children.append(child_1)
            children.append(child_2)
        print("after crossover:\n", parents)

        return children

    def mutation(self, children):

        print(35 * "=", "single point mutation", 35 * "=")
        print("before mutation:\n", children)

        for index in range(len(children)):
            # mutation judge
            mutation_op = np.random.choice([True, False], 1, p=[self.mutation_rate, 1 - self.mutation_rate])[0]
            if mutation_op:
                # select mutation point based on random
                position_to_mutate = np.random.randint(len(children[index]))
                space_list = self.search_space[self.stack_gcn_architecture[position_to_mutate]]
                children[index][position_to_mutate] = np.random.randint(0, len(space_list))
        print("after mutation:\n", children)

        return children

    def updating(self, children_embedding, children, children_fitness):

        print(35*"=", "population updating", 35*"=")

        _, top_fitness = utils.top_population_select(self.population,
                                                     self.fitness,
                                                     top_k=self.update_threshold)
        threshold = np.mean(top_fitness)

        # new child updates into population if fitness bigger than threshold
        index = 0
        for fitness in children_fitness:
            if fitness > threshold:
                self.fitness.append(fitness)
                self.population.append(children[index])
                self.population_embedding.append(children_embedding[index])
                index += 1
            else:
                index += 1

class Search(object):

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.search_space = search_space

    def search_operator(self):

        print(35 * "=", "genetic search start", 35 * "=")

        time_start = time.time()

        # Naive　Genetic Searcher Initialize
        searcher = GeneticSearch(self.search_space,
                                 int(self.search_parameter["initial_population_scale"]),
                                 int(self.search_parameter["parent_scale"]),
                                 float(self.search_parameter["crossover_rate"]),
                                 float(self.search_parameter["mutation_rate"]),
                                 int(self.search_parameter["update_threshold"]))

        # Parallel Operator Module initialize
        parallel_estimation = ParallelOperater(self.data, self.gnn_parameter)

        # parallel fitness estimation
        searcher.fitness = parallel_estimation.estimation(searcher.population)

        initial_time = time.time() - time_start

        # initial time record
        path = os.path.split(os.path.realpath(__file__))[0][:-34] + "/logger/genetic_logger/"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save_initial(path, self.data.data_name + "_initial_time.txt", initial_time)

        print(35*"=", "search start", 35*"=")

        time_search_list = []
        epoch = []

        # naive genetic search_algorithm
        for i in range(int(self.search_parameter["search_epoch"])):

            time_start = time.time()

            print(35*"=", "the ", str(i+1), " th search epoch", 35*"=")

            children_embedding = searcher.search()
            children = []
            for gnn_architecture_embedding in children_embedding:
                children.append(utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space.space_getter(),
                                                                        self.search_space.stack_gcn_architecture))

            # parallel estimation
            children_fitness = parallel_estimation.estimation(children)

            # population updating
            searcher.updating(children_embedding, children, children_fitness)

            # check population
            if len(searcher.population) >= 10:
                top_population, top_fitness = utils.top_population_select(searcher.population, searcher.fitness, 10)
            else:
                top_population, top_fitness = utils.top_population_select(searcher.population, searcher.fitness, 1)

            # model architecture / fitness record
            path = os.path.split(os.path.realpath(__file__))[0][:-34] + "/logger/genetic_logger/"
            if not os.path.exists(path):
                os.makedirs(path)
            utils.experiment_data_record(path,
                                         self.data.data_name + "_search_epoch_" + str(i+1) + ".txt",
                                         top_population,
                                         top_fitness)

            time_search_list.append(time.time()-time_start)
            epoch.append(i+1)
            print("pop:", len(searcher.population))

        index = top_population.index(max(top_population))
        best_val_architecture = top_population[index]
        best_performance = max(top_fitness)
        print("the best architecture:\n", best_val_architecture)
        print("the best performance:\n", best_performance)

        # search_algorithm time record
        path = os.path.split(os.path.realpath(__file__))[0][:-34] + "/logger/genetic_logger/"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   self.data.data_name + "_search_time.txt",
                                   epoch,
                                   time_search_list)

if __name__=="__main__":

    # ParallelConfig(True)
    ParallelConfig(False)
    graph = CiteNetwork("cora")

    search_parameter = {"initial_population_scale": "10",
                        "parent_scale": "6",
                        "crossover_rate": "1.0",
                        "mutation_rate": "0.02",
                        "update_threshold": "5",
                        "search_epoch": "5"}

    gnn_parameter = {"gnn_type": "stack_gcn",
                     "downstream_task_type": "transductive_node_classification",
                     "gnn_drop_out": "0.6",
                     "train_epoch": "10",
                     "early_stop": "False",
                     "early_stop_patience": "10",
                     "opt_type": "adam",
                     "opt_type_dict": "{\"learning_rate\": 0.005, \"l2_regularization_strength\": 0.0005}",
                     "loss_type": "nll_loss",
                     "val_evaluator_type": "accuracy",
                     "test_evaluator_type": "[\"accuracy\", \"precision\", \"recall\", \"f1_value\"]"}

    search_space = SearchSpace(gnn_layers="2")
    instance = Search(graph, search_parameter, gnn_parameter, search_space)
    instance.search_operator()
