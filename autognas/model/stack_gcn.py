import torch
from autognas.model.stack_gcn_encoder.gcn_encoder import GcnEncoder
from autognas.model.logger import gnn_architecture_performance_save,\
                                  test_performance_save,\
                                  model_save
from autognas.dynamic_configuration import optimizer_getter,  \
                                           loss_getter, \
                                           evaluator_getter, \
                                           downstream_task_model_getter

from autognas.datasets.util_cite_network import CiteNetwork  # for unit test

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
            10. num_features, 11.data_name
        gnn_architecture: list
            the stack gcn architecture describe
            for example: ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
        gnn_drop_out: float
            the drop out rate for stack gcn model for every layer
        train_epoch: int
            the model train epoch
        early_stop: bool
            controlling the whether use early stop mechanism in the model training process
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
                 downstream_task_type="transductive_node_classification",
                 gnn_architecture=['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh'],
                 gnn_drop_out=0.6,
                 train_epoch=300,
                 early_stop=False,
                 early_stop_patience=10,
                 opt_type="adam",
                 opt_parameter_dict={"learning_rate": 0.005, "l2_regularization_strength": 0.0005},
                 loss_type="nll_loss",
                 val_evaluator_type="accuracy",
                 test_evaluator_type=["accuracy", "precision", "recall", "f1_value"]):

        self.graph_data = graph_data
        self.downstream_task_type = downstream_task_type
        self.gnn_architecture = gnn_architecture
        self.gnn_drop_out = gnn_drop_out
        self.train_epoch = train_epoch
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.opt_type = opt_type
        self.opt_parameter_dict = opt_parameter_dict
        self.loss_type = loss_type
        self.val_evaluator_type = val_evaluator_type
        self.test_evaluator_type = test_evaluator_type

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gnn_model = GcnEncoder(self.gnn_architecture,
                                    self.graph_data.num_features,
                                    dropout=self.gnn_drop_out).to(self.device)

        self.optimizer = optimizer_getter(self.opt_type,
                                          self.gnn_model,
                                          self.opt_parameter_dict)

        self.loss = loss_getter(self.loss_type)

        self.val_evaluator = evaluator_getter(self.val_evaluator_type)

        self.downstream_task_model = downstream_task_model_getter(self.downstream_task_type,
                                                                  int(self.gnn_architecture[-2]),
                                                                  self.graph_data).to(self.device)

    def fit(self):

        val_loss_list = []
        early_stop_flag = False

        for epoch in range(1, self.train_epoch + 1):
            self.gnn_model.train()
            node_embedding = self.gnn_model(self.graph_data.train_x,
                                            self.graph_data.train_edge_index)
            train_predict_y = self.downstream_task_model(node_embedding, mode="train")
            train_loss = self.loss(train_predict_y,
                                   self.graph_data.train_y)
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            train_loss = train_loss.item()

            self.gnn_model.eval()
            node_embedding = self.gnn_model(self.graph_data.val_x,
                                            self.graph_data.val_edge_index)
            val_predict_y = self.downstream_task_model(node_embedding, mode="val")
            val_loss = self.loss(val_predict_y,
                                 self.graph_data.val_y)
            val_loss = val_loss.item()
            val_loss_list.append(val_loss)

            if self.early_stop:
                early_stop_flag = self.early_stopping(val_loss_list,
                                                      self.early_stop_patience)

            if early_stop_flag:
                print("early stopping epoch:", epoch, "\n")
                break

        self.gnn_model.eval()
        train_performance = self.val_evaluator.function(train_predict_y, self.graph_data.train_y)
        val_performance = self.val_evaluator.function(val_predict_y, self.graph_data.val_y)

        print("\n" + "stack gcn architecture:\t" + str(self.gnn_architecture) + "\n" +
              "train loss:\t" + str(train_loss) + "\n" +
              "train " + str(self.val_evaluator_type) + ":" + "\t" + str(train_performance) + "\n"
              "val loss:\t" + str(val_loss) + "\n" +
              "val " + str(self.val_evaluator_type) + ":" + "\t" + str(val_performance))

        gnn_architecture_performance_save(self.gnn_architecture, val_performance, self.graph_data.data_name)

        return val_performance

    def evaluate(self, model_num=0):

        self.gnn_model.eval()
        node_embedding = self.gnn_model(self.graph_data.test_x, self.graph_data.test_edge_index)
        test_predict_y = self.downstream_task_model(node_embedding, mode="test")
        test_loss = self.loss(test_predict_y,
                              self.graph_data.test_y)

        print("test gnn architecture:\t", str(self.gnn_architecture))
        test_performance_dict = {"test loss": test_loss.item()}
        print("test loss:\t" + str(test_loss.item()))

        for evaluator_type in self.test_evaluator_type:
            test_evaluator = evaluator_getter(evaluator_type)
            test_performance = test_evaluator.function(test_predict_y, self.graph_data.test_y)
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
    graph = CiteNetwork("cora")
    model = StackGcn(graph)
    performance = model.fit()
    model.evaluate()
