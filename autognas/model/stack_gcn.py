import torch
from autognas.model.util import batch_util
from autognas.model.stack_gcn_encoder.gcn_encoder import GcnEncoder
from autognas.model.logger import gnn_architecture_performance_save,\
                                  test_performance_save,\
                                  model_save
from autognas.dynamic_configuration import optimizer_getter,  \
                                           loss_getter, \
                                           evaluator_getter, \
                                           downstream_task_model_getter
from autognas.datasets.planetoid import Planetoid

class StackGcn(object):
    """
    Realizing stack GCN  model initializing, downstream task model initializing,
    model training validating and testing based on graph data and stack gcn architecture.

    Args:
        graph_data: graph data obj
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        gnn_architecture: list
            the stack gcn architecture describe
            for example: ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
        train_batch_size: int
            the batch size of train dataset
        val_batch_size: int
            the batch size of validation dataset
        test_batch_size: int
            the batch size of test dataset
        gnn_drop_out: float
            the drop out rate for stack gcn model for every layer
        train_epoch: int
            the model train epoch for validation
        train_epoch_test: int
            the model train epoch for testing
        bias: bool
            controlling whether add bias to the GNN model
        early_stop: bool
            controlling  whether use early stop mechanism in the model training process
        early_stop_patience: int
            controlling validation loss comparing cycle of the early stop mechanism
        opt_type: str
            the optimization function type for the model
        opt_parameter_dict: dict
            the hyper-parameter of selected optimizer
        loss_type: str
            the loss function type for the model
        val_evaluator_type: str
            the validation evaluating metric in the model training process
        test_evaluator_type: list
            the testing evaluating metric in the model testing process

    Returns:
        val_performance: float
            the validation result
    """

    def __init__(self,
                 graph_data,
                 downstream_task_type="node_classification",
                 train_batch_size=1,
                 val_batch_size=1,
                 test_batch_size=1,
                 gnn_architecture=['gcn', 'sum',  1, 128, 'relu', 'gcn', 'sum', 1, 64, 'linear'],
                 gnn_drop_out=0.6,
                 train_epoch=100,
                 train_epoch_test=100,
                 bias=True,
                 early_stop=False,
                 early_stop_patience=10,
                 opt_type="adam",
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="nll_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["accuracy", "precision", "recall", "f1_value"]):

        self.graph_data = graph_data
        self.downstream_task_type = downstream_task_type
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.gnn_architecture = gnn_architecture
        self.gnn_drop_out = gnn_drop_out
        self.train_epoch = train_epoch
        self.train_epoch_test = train_epoch_test
        self.bias = bias
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.opt_type = opt_type
        self.opt_parameter_dict = opt_parameter_dict
        self.loss_type = loss_type
        self.val_evaluator_type = val_evaluator_type
        self.test_evaluator_type = test_evaluator_type
        self.train_batch_id = 0
        self.val_batch_id = 0
        self.test_batch_id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gnn_model = GcnEncoder(self.gnn_architecture,
                                    self.graph_data.num_features,
                                    dropout=self.gnn_drop_out,
                                    bias=self.bias).to(self.device)

        self.optimizer = optimizer_getter(self.opt_type,
                                          self.gnn_model,
                                          self.opt_parameter_dict)

        self.loss = loss_getter(self.loss_type)

        self.val_evaluator = evaluator_getter(self.val_evaluator_type)

        self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

    def fit(self):
        # training on the training dataset based on training batch dataset
        batch_train_x_list, \
        batch_train_edge_index_list, \
        batch_train_y_list, \
        batch_train_x_index_list = batch_util(self.train_batch_size,
                                              self.graph_data.train_x,
                                              self.graph_data.train_edge_index,
                                              self.graph_data.train_y)

        for epoch in range(1, self.train_epoch + 1):

            self.train_batch_id = 0
            self.gnn_model.train()
            train_predict_y_list = []
            train_y_list = []
            one_epoch_train_loss_list = []

            for train_x, train_edge_index, train_y in zip(batch_train_x_list,
                                                          batch_train_edge_index_list,
                                                          batch_train_y_list):

                node_embedding_matrix = self.gnn_model(train_x, train_edge_index)
                train_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                             batch_train_x_index_list[self.train_batch_id],
                                                             mode="train")

                self.train_batch_id += 1
                train_loss = self.loss.function(train_predict_y, train_y)
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()

                one_epoch_train_loss_list.append(train_loss.item())
                train_predict_y_list.append(train_predict_y)
                train_y_list.append(train_y)

            train_loss = sum(one_epoch_train_loss_list)

            # validating on the validation dataset based on validation batch dataset in the training process
            self.gnn_model.eval()
            self.val_batch_id = 0
            batch_val_loss_list = []
            val_predict_y_list = []
            val_y_list = []
            val_loss_list = []
            early_stop_flag = False

            batch_val_x_list, \
            batch_val_edge_index_list, \
            batch_val_y_list, \
            batch_val_x_index_list = batch_util(self.val_batch_size,
                                                self.graph_data.val_x,
                                                self.graph_data.val_edge_index,
                                                self.graph_data.val_y)

            for val_x, val_edge_index, val_y in zip(batch_val_x_list,
                                                    batch_val_edge_index_list,
                                                    batch_val_y_list):

                node_embedding_matrix = self.gnn_model(val_x,
                                                       val_edge_index)

                val_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                           batch_val_x_index_list[self.val_batch_id],
                                                           mode="val")

                self.val_batch_id += 1
                val_loss = self.loss.function(val_predict_y, val_y)
                batch_val_loss_list.append(val_loss.item())
                val_predict_y_list.append(val_predict_y)
                val_y_list.append(val_y)

            val_loss = sum(batch_val_loss_list)
            val_loss_list.append(val_loss)

            torch.cuda.empty_cache()

            if self.early_stop:
                early_stop_flag = self.early_stopping(val_loss_list,
                                                      self.early_stop_patience)

            if early_stop_flag:
                print("early stopping epoch:", epoch, "\n")
                break

        self.gnn_model.eval()
        train_performance = 0
        val_performance = 0
        for train_predict_y, train_y in zip(train_predict_y_list,
                                            train_y_list):

            train_performance += self.val_evaluator.function(train_predict_y, train_y)

        for val_predict_y, val_y in zip(val_predict_y_list,
                                        val_y_list):

            val_performance += self.val_evaluator.function(val_predict_y, val_y)

        train_performance = train_performance/self.train_batch_id

        val_performance = val_performance/self.val_batch_id

        print("\n" + "stack gcn architecture:\t" + str(self.gnn_architecture) + "\n" +
              "train loss:\t" + str(train_loss) + "\n" +
              "train " + str(self.val_evaluator_type) + ":" + "\t" + str(train_performance) + "\n"
              "val loss:\t" + str(val_loss) + "\n" +
              "val " + str(self.val_evaluator_type) + ":" + "\t" + str(val_performance))

        gnn_architecture_performance_save(self.gnn_architecture, val_performance, self.graph_data.data_name)


        return val_performance

    def evaluate(self, model_num=0):

        # train the model from the scratch based on train_epoch_test
        print(25*"#", "testing, train from the scratch based on train_epoch_test", 25*"#")
        self.train_epoch = self.train_epoch_test
        self.fit()

        # testing on the test dataset based on test batch dataset
        self.gnn_model.eval()
        test_predict_y_list = []
        test_loss_list = []
        test_y_list = []

        batch_test_x_list, \
        batch_test_edge_index_list, \
        batch_test_y_list, \
        batch_test_x_index_list = batch_util(self.test_batch_size,
                                             self.graph_data.test_x,
                                             self.graph_data.test_edge_index,
                                             self.graph_data.test_y)

        self.gnn_model.eval()

        for test_x, test_edge_index, test_y in zip(batch_test_x_list,
                                                   batch_test_edge_index_list,
                                                   batch_test_y_list):
            node_embedding_matrix = self.gnn_model(test_x, test_edge_index)
            test_predict_y = self.downstream_task_model(node_embedding_matrix,
                                                        batch_test_x_index_list[self.test_batch_id],
                                                        mode="test")
            self.test_batch_id += 1
            test_loss = self.loss.function(test_predict_y, test_y)
            test_predict_y_list.append(test_predict_y)
            test_y_list.append(test_y)
            test_loss_list.append(test_loss.item())

        test_loss = sum(test_loss_list)

        print("test gnn architecture:\t", str(self.gnn_architecture))
        test_performance_dict = {"test loss": test_loss}
        print("test loss:\t" + str(test_loss))

        for evaluator_type in self.test_evaluator_type:
            test_performance = 0
            for test_predict_y, test_y in zip(test_predict_y_list, test_y_list):
                test_evaluator = evaluator_getter(evaluator_type)
                test_performance = test_performance + test_evaluator.function(test_predict_y, test_y)
            test_performance = test_performance/self.test_batch_id
            test_performance_dict[evaluator_type] = str(test_performance)
            print("test " + evaluator_type + ":" + "\t" + str(test_performance))

        hyperparameter_dict = {"gnn_drop_out": self.gnn_drop_out,
                               "train_epoch": self.train_epoch,
                               "early_stop": self.early_stop,
                               "early_stop_patience": self.early_stop_patience,
                               "optimizer": self.opt_type,
                               "opt_parameter_dict": self.opt_parameter_dict,
                               "loss_type": self.loss_type}

        test_performance_save(self.gnn_architecture,
                              test_performance_dict,
                              hyperparameter_dict,
                              self.graph_data.data_name)

        model_save(self.gnn_model, self.optimizer, self.graph_data.data_name, model_num)

    def early_stopping(self, val_loss_list, stop_patience):

        if len(val_loss_list) < stop_patience:
            return False

        if val_loss_list[-stop_patience:][0] > val_loss_list[-1]:
            return True

        else:
            return False

if __name__=="__main__":

    # node classification test
    graph = Planetoid("cora").data

    model = StackGcn(graph,
                     train_epoch=10,
                     train_epoch_test=100,
                     gnn_architecture=['gcn', 'softmax_sum',  1, 128, 'relu', 'gat', 'sum', 1, 64, 'linear'],
                     downstream_task_type="node_classification",
                     val_evaluator_type="accuracy",
                     test_evaluator_type=["accuracy", "precision", "recall", "f1_value"])
    model.evaluate()

    # graph classification test

    # link prediction test
    # graph = Planetoid("cora_lp",
    #                   train_splits=0.85,
    #                   val_splits=0.05,
    #                   shuffle_flag=False).data
    #
    # model = StackGcn(graph,
    #                  downstream_task_type="link_prediction",
    #                  gnn_architecture=['gcn', 'sum', 1, 128, 'relu', 'gcn', 'sum', 1, 64, 'linear'],
    #                  train_epoch=10,
    #                  train_epoch_test=50,
    #                  gnn_drop_out=0.3,
    #                  opt_parameter_dict={"learning_rate": 0.01, "l2_regularization_strength": 0.0005},
    #                  loss_type="binary_cross_entropy",
    #                  val_evaluator_type="roc_auc_score",
    #                  test_evaluator_type=["roc_auc_score"]
    #                  )
    # model.evaluate()

    # graph = Planetoid("AIDS").data
    #
    # model = StackGcn(graph,
    #                  gnn_architecture=['linear', 'mean', 8, 256, 'sigmoid', 'cos', 'max', 4, 128, 'softplus'],
    #                  downstream_task_type="graph_classification",
    #                  train_batch_size=50,
    #                  val_batch_size=10,
    #                  test_batch_size=10,
    #                  train_epoch=10,
    #                  train_epoch_test=100,
    #                  gnn_drop_out=0.5,
    #                  loss_type="cross_entropy_loss",
    #                  val_evaluator_type="accuracy",
    #                  test_evaluator_type=["accuracy", "precision", "recall", "f1_value"])
    # model.evaluate()