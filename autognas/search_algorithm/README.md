## Search algorithm user-defined specification

- Users only need to define their own search algorithm according to the following template in the **user-defined area**, and then name the user-defined script **search_algorithm.py**. Next put the **search_algorithm.py** into a file which the file name is the search algorithm name for example graphpas. Finally, put the file into this path: **autognas/search_algorithm/**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**
```python

# import what you need to import python package

from autognas.parallel import ParallelOperater # for parallel estimation

class UserCustomSearch(object):
    '''
    This class implement the sampling new gnn architectures 
    
    Args:
        user_defined: user can defined what they need to use variable
        
    Returns:
        gnn_architecture_list: list
            the new gnn architectures sampled by 
            user-defind search algorithm as a list
    '''

    def __init__(self,user_defined):
        
        # User-defined  area
        
        self.gnn_architecture_list = []

    def search(self,user_defined):

        # User-defined  area

        return self.gnn_architecture_list

class Search(object):
    '''
    This class implement user-defind search algortihm logic
    including sampled gnn architectures parallel estimation
    
    Args:
        data: graph object
            the target graph data object including required attributes:
            1.train_x, 2.train_y, 3.train_edge_index
            4.val_x, 5.val_y, 6.val_edge_index
            7.test_x, 8.test_y, 9.test_edge_index
            10. num_features, 11.num_labels, 12.data_name
        
        search_parameter:dict
            the configuration dict of search algorithm including required 2 key:value
              key                    value
            1.search_algorithm_type : user-defind search algorithm name
            2.test_gnn_num          : the number of finall test gnn model derived  
                                      from the sampled gnn architectures
            
        gnn_parameter:dict
            the configuration dict of gnn model training including optimal key:value
                key                   value
                gnn_type            : stack_gcn
                gnn_layers          : 2
                downstream_task_type: node_classification
                train_batch_size    : 1
                val_batch_size      : 1
                test_batch_size     : 1
                gnn_drop_out        : 0.6
                train_epoch         : 10
                early_stop          : False
                early_stop_patience : 10
                opt_type            : adam
                opt_type_dict       : {"learning_rate": 0.005, "l2_regularization_strength": 0.0005}
                loss_type           : nll_loss
                val_evaluator_type  : accuracy
                test_evaluator_type : ["accuracy", "precision", "recall", "f1_value"]
         
         search_space:dict
             the search space dict, the key represents the gnn architecture compenent name
             the value represents the corresponding the component values
           
         Returns:
              None
            
    '''

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.search_space = search_space
        
        # User-defined  area

    def search_operator(self,user_defined):
        
        # User-defined  area
        
        searcher = UserCustomSearch(user_defind)
        
        # parallel estimation object initializing         
        parallel_estimation = ParallelOperater(self.data, self.gnn_parameter)

        gnn_architecture_list = searcher.search(user_defind)
        
        # parallell estimation 
        result = parallel_estimation.estimation(gnn_architecture_list)

```
