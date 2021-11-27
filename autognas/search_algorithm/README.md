
## Search Algorithm user-defined specification

- Users only need to define their own search algorithm according to the following template in the **user-defined area** , and then put the user-defined script into a file, and file name is your search algorithm name. putting this file into **autognas/search_algorithm/**. the Auto-GNAS will automatically load it. 

- **Warning:don't modify other parts of the template to avoid automatic loading failure!**

```python

from autognas.parallel import ParallelOperater, ParallelConfig
# import what you need to import python package

class UserDefinedSearch(object):

    def __init__(self,search_space,user_defined_parameter):
        
        # get seach space component dict {component:value}
        self.search_space = search_space.space_getter()
        # get stack gnn architecture list 
        # ['attention','aggregation','multi_heads', 'hidden_dimension','activation']
        self.stack_gcn_architecture = search_space.stack_gcn_architecture
        # initialize the gnn_architecture_list
        self.gnn_architecture_list = []
        
        # User-defined  area

    def search(self):

        # User-defined  area

        return self.gnn_architecture_list

class Search(object):

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.data = data                         # graph data object
        self.search_parameter = search_parameter # search parameter dict
        self.gnn_parameter = gnn_parameter       # gnn train/test parameter dict
        self.search_space = search_space         # search space component dict, stack gnn architecture list

    def search_operator(self):
        
        # User-defined  area

        # Parallel Operator object initialize
        parallel_estimation = ParallelOperater(self.data, self.gnn_parameter)
        # initialize UserDefinedSearch object
        searcher = self.UserDefinedSearch(search_space,user_defined_parameter)
        # get sampled gnn architectures list from UserDefinedSearch 
        gnn_architecture_list = searcher.search()
        # get parallel estimation results for sampled gnn architectures
        result = parallel_estimation.estimation(gnn_architecture_list)

       
```
