##  ini configuration script user-defined specification

- Users only need to define their own ini configuration file according to the following template in the **user-defined area** , and then ensure you can read the config script in your main.py. 

- **Attention:in your ini script you need include necessary information that have been marked in the template**

- **Warning:don't modify other parts of the template to avoid AutoGNAS failure!**

```python
[search_parameter]
search_algorithm_type = user-defined search name       # necessary
test_gnn_num = the final test model number             # optimal
# user-defined area


[gnn_parameter]
gnn_type = stack_gcn                                   # optimal
# training gnn type in AutoGNAS and AutoGNAS supports "stack gcn" in current version
gnn_layers = 2                                         # optimal
downstream_task_type = node_classification             # optimal
train_batch_size = 1                                   # optimal
val_batch_size = 1                                     # optimal
test_batch_size = 1                                    # optimal
gnn_drop_out = 0.6                                     # optimal
train_epoch = 10                                       # optimal
early_stop = False                                     # optimal
early_stop_patience = 10                               # optimal
opt_type = adam                                        # optimal
opt_type_dict = {"learning_rate": 0.005, "l2_regularization_strength": 0.0005}  # optimal
loss_type = nll_loss
val_evaluator_type = accuracy                          # optimal
test_evaluator_type = ["accuracy", "precision", "recall", "f1_value"]           # optimal
```
