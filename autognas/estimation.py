from autognas.model.stack_gcn import StackGcn
from autognas.datasets.planetoid import Planetoid

class Estimation(object):
    """
    Realizing the gnn parameter parsing and checking
    for different gnn architecture type, validating
    and testing the model performance

    Args:
        gnn_architecture: list
            the gnn architecture describe
            for example, the stack gcn architecture describe:
            ['gcn', 'sum',  1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
        data: graph data object
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        gnn_parameter: dict
            the gnn model training validating testing config dict

    Returns:
        performance: float
            the validation performance of model
    """

    def __init__(self,
                 gnn_architecture,
                 data,
                 gnn_parameter):

        if not isinstance(gnn_architecture, list):
            raise Exception("gnn_architecture Class Wrong, require list Class ", "but input Class:",
                            type(gnn_architecture))

        if not isinstance(gnn_parameter, dict):
            raise Exception("gnn_parameter Class Wrong, require dict Class ", "but input Class:",
                            type(gnn_parameter))

        self.gnn_architecture = gnn_architecture
        self.data = data

        if "gnn_type" in gnn_parameter["gnn_type"]:
            self.gnn_type = gnn_parameter["gnn_type"]
        else:
            self.gnn_type = "stack_gcn"

        self.gnn_parameter = gnn_parameter

    def get_performance(self):

        if self.gnn_type == "stack_gcn":

            downstream_task_type = "node_classification"
            train_batch_size = 1
            val_batch_size = 1
            test_batch_size = 1
            gnn_drop_out = 0.6
            train_epoch = 100
            bias = True
            early_stop = False
            early_stop_patience = 10
            opt_type = "adam"
            opt_parameter_dict = {"learning_rate": 0.005, "l2_regularization_strength": 0.0005}
            loss_type = "nll_loss"
            val_evaluator_type = "accuracy"
            test_evaluator_type = ["accuracy", "precision", "recall", "f1_value"]

            if "downstream_task_type" in self.gnn_parameter:
                downstream_task_type = self.gnn_parameter["downstream_task_type"]
            if "train_batch_size" in self.gnn_parameter:
                train_batch_size = eval(self.gnn_parameter["train_batch_size"])
            if "val_batch_size" in self.gnn_parameter:
                val_batch_size = eval(self.gnn_parameter["val_batch_size"])
            if "test_batch_size" in self.gnn_parameter:
                test_batch_size = eval(self.gnn_parameter["test_batch_size"])
            if "gnn_drop_out" in self.gnn_parameter:
                gnn_drop_out = eval(self.gnn_parameter["gnn_drop_out"])
            if "train_epoch" in self.gnn_parameter:
                train_epoch = eval(self.gnn_parameter["train_epoch"])
            if "bias" in self.gnn_parameter:
                bias = eval(self.gnn_parameter["bias"])
            if "early_stop" in self.gnn_parameter:
                early_stop = eval(self.gnn_parameter["early_stop"])
            if "early_num" in self.gnn_parameter:
                early_stop_patience = eval(self.gnn_parameter["early_stop_patience"])
            if "opt_type" in self.gnn_parameter:
                opt_type = self.gnn_parameter["opt_type"]
            if "opt_parameter_dict" in self.gnn_parameter:
                opt_parameter_dict = eval(self.gnn_parameter["opt_parameter_dict"])
            if "loss_type" in self.gnn_parameter:
                loss_type = self.gnn_parameter["loss_type"]
            if "val_evaluator_type" in self.gnn_parameter:
                val_evaluator_type = self.gnn_parameter["val_evaluator_type"]
            if "test_evaluator_type" in self.gnn_parameter:
                test_evaluator_type = eval(self.gnn_parameter["test_evaluator_type"])

            if not isinstance(downstream_task_type, str):
                raise Exception("downstream_task_type Class Wrong, require str Class ", "but input Class: ",
                                type(downstream_task_type))

            if not isinstance(train_batch_size, int):
                raise Exception("train_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(train_batch_size))

            if not isinstance(val_batch_size, int):
                raise Exception("val_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(val_batch_size))

            if not isinstance(test_batch_size, int):
                raise Exception("test_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(test_batch_size))

            if not isinstance(gnn_drop_out, float):
                raise Exception("gnn_drop_out Class Wrong, require float Class ", "but input Class: ",
                                type(gnn_drop_out))

            if not isinstance(train_epoch, int):
                raise Exception("train_epoch Class Wrong, require int Class ", "but input Class: ",
                                type(train_epoch))

            if not isinstance(bias, bool):
                raise Exception("bias Class Wrong, require bool  Class ", "but input Class: ",
                                type(bias))

            if not isinstance(early_stop, bool):
                raise Exception("early_stop Class Wrong, require bool Class ", "but input Class: ",
                                type(early_stop))

            if not isinstance(early_stop_patience, int):
                raise Exception("early_stop_patience Class Wrong, require int Class ", "but input Class: ",
                                type(early_stop_patience))

            if not isinstance(opt_type, str):
                raise Exception("opt_type Class Wrong, require str Class ", "but input Class: ",
                                type(opt_type))

            if not isinstance(opt_parameter_dict, dict):
                raise Exception("opt_parameter_dict Class Wrong, require dict Class ", "but input Class: ",
                                type(opt_parameter_dict))

            if not isinstance(loss_type, str):
                raise Exception("loss_type Class Wrong, require str Class ", "but input Class: ",
                                type(loss_type))

            if not isinstance(val_evaluator_type, str):
                raise Exception("val_evaluator_type Class Wrong, require str Class ", "but input Class: ",
                                type(val_evaluator_type))

            if not isinstance(test_evaluator_type, list):
                raise Exception("test_evaluator_type Class Wrong, require list Class ", "but input Class: ",
                                type(test_evaluator_type))

            model = StackGcn(graph_data=self.data,
                             downstream_task_type=downstream_task_type,
                             train_batch_size=train_batch_size,
                             val_batch_size=val_batch_size,
                             test_batch_size=test_batch_size,
                             gnn_architecture=self.gnn_architecture,
                             gnn_drop_out=gnn_drop_out,
                             train_epoch=train_epoch,
                             bias=bias,
                             early_stop=early_stop,
                             early_stop_patience=early_stop_patience,
                             opt_type=opt_type,
                             opt_parameter_dict=opt_parameter_dict,
                             loss_type=loss_type,
                             val_evaluator_type=val_evaluator_type,
                             test_evaluator_type=test_evaluator_type)

            performance = model.fit()

            return performance

        else:
            raise Exception("Wrong gnn type")

    def get_test_result(self, model_num=0):

        if self.gnn_type == "stack_gcn":

            downstream_task_type = "node_classification"
            train_batch_size = 1
            val_batch_size = 1
            test_batch_size = 1
            gnn_drop_out = 0.6
            train_epoch = 100
            bias = True
            early_stop = False
            early_stop_patience = 10
            opt_type = "adam"
            opt_parameter_dict = {"learning_rate": 0.005, "l2_regularization_strength": 0.0005}
            loss_type = "nll_loss"
            val_evaluator_type = "accuracy"
            test_evaluator_type = ["accuracy", "precision", "recall", "f1_value"]

            if "downstream_task_type" in self.gnn_parameter:
                downstream_task_type = self.gnn_parameter["downstream_task_type"]
            if "train_batch_size" in self.gnn_parameter:
                train_batch_size = eval(self.gnn_parameter["train_batch_size"])
            if "val_batch_size" in self.gnn_parameter:
                val_batch_size = eval(self.gnn_parameter["val_batch_size"])
            if "test_batch_size" in self.gnn_parameter:
                test_batch_size = eval(self.gnn_parameter["test_batch_size"])
            if "gnn_drop_out" in self.gnn_parameter:
                gnn_drop_out = eval(self.gnn_parameter["gnn_drop_out"])
            if "train_epoch" in self.gnn_parameter:
                train_epoch = eval(self.gnn_parameter["train_epoch"])
            if "bias" in self.gnn_parameter:
                bias = eval(self.gnn_parameter["bias"])
            if "early_stop" in self.gnn_parameter:
                early_stop = eval(self.gnn_parameter["early_stop"])
            if "early_num" in self.gnn_parameter:
                early_stop_patience = eval(self.gnn_parameter["early_stop_patience"])
            if "opt_type" in self.gnn_parameter:
                opt_type = self.gnn_parameter["opt_type"]
            if "opt_parameter_dict" in self.gnn_parameter:
                opt_parameter_dict = eval(self.gnn_parameter["opt_parameter_dict"])
            if "loss_type" in self.gnn_parameter:
                loss_type = self.gnn_parameter["loss_type"]
            if "val_evaluator_type" in self.gnn_parameter:
                val_evaluator_type = self.gnn_parameter["val_evaluator_type"]
            if "test_evaluator_type" in self.gnn_parameter:
                test_evaluator_type = eval(self.gnn_parameter["test_evaluator_type"])

            if not isinstance(downstream_task_type, str):
                raise Exception("downstream_task_type Class Wrong, require str Class ", "but input Class: ",
                                type(downstream_task_type))

            if not isinstance(train_batch_size, int):
                raise Exception("train_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(train_batch_size))

            if not isinstance(val_batch_size, int):
                raise Exception("val_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(val_batch_size))

            if not isinstance(test_batch_size, int):
                raise Exception("test_batch_size Class Wrong, require int Class ", "but input Class: ",
                                type(test_batch_size))

            if not isinstance(gnn_drop_out, float):
                raise Exception("gnn_drop_out Class Wrong, require float Class ", "but input Class: ",
                                type(gnn_drop_out))

            if not isinstance(train_epoch, int):
                raise Exception("train_epoch Class Wrong, require int Class ", "but input Class: ",
                                type(train_epoch))

            if not isinstance(bias, bool):
                raise Exception("bias Class Wrong, require bool  Class ", "but input Class: ",
                                type(bias))

            if not isinstance(early_stop, bool):
                raise Exception("early_stop Class Wrong, require bool Class ", "but input Class: ",
                                type(early_stop))

            if not isinstance(early_stop_patience, int):
                raise Exception("early_stop_patience Class Wrong, require int Class ", "but input Class: ",
                                type(early_stop_patience))

            if not isinstance(opt_type, str):
                raise Exception("opt_type Class Wrong, require str Class ", "but input Class: ",
                                type(opt_type))

            if not isinstance(opt_parameter_dict, dict):
                raise Exception("opt_parameter_dict Class Wrong, require dict Class ", "but input Class: ",
                                type(opt_parameter_dict))

            if not isinstance(loss_type, str):
                raise Exception("loss_type Class Wrong, require str Class ", "but input Class: ",
                                type(loss_type))

            if not isinstance(val_evaluator_type, str):
                raise Exception("val_evaluator_type Class Wrong, require str Class ", "but input Class: ",
                                type(val_evaluator_type))

            if not isinstance(test_evaluator_type, list):
                raise Exception("test_evaluator_type Class Wrong, require list Class ", "but input Class: ",
                                type(test_evaluator_type))

            model = StackGcn(graph_data=self.data,
                             downstream_task_type=downstream_task_type,
                             train_batch_size=train_batch_size,
                             val_batch_size=val_batch_size,
                             test_batch_size=test_batch_size,
                             gnn_architecture=self.gnn_architecture,
                             gnn_drop_out=gnn_drop_out,
                             train_epoch=train_epoch,
                             bias=bias,
                             early_stop=early_stop,
                             early_stop_patience=early_stop_patience,
                             opt_type=opt_type,
                             opt_parameter_dict=opt_parameter_dict,
                             loss_type=loss_type,
                             val_evaluator_type=val_evaluator_type,
                             test_evaluator_type=test_evaluator_type)

            model.fit()
            model.evaluate(model_num)

        else:
            raise Exception("Wrong gnn type")

if __name__=="__main__":

    graph = Planetoid("cora").data
    gnn_architecture = ['gcn', 'sum', 1, 64, 'tanh', 'gcn', 'sum', 1, 64, 'tanh']
    estimator = Estimation(gnn_architecture, graph, {"gnn_type": "stack_gcn"})
    performance = estimator.get_performance()
    estimator.get_test_result()
