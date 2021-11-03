from autognas.dynamic_configuration import data_util_class_getter

class Planetoid(object):

    def __init__(self,
                 data_name="cora",
                 train_splits=None,
                 val_splits=None,
                 shuffle_flag=False,
                 random_seed=None):

        self.data = self.__get_data(data_name,
                                    train_splits,
                                    val_splits,
                                    shuffle_flag,
                                    random_seed)

    def __get_data(self,
                   data_name,
                   train_splits,
                   val_splits,
                   shuffle_flag,
                   random_seed):

        data_util_dict = data_util_class_getter()

        for data_util_obj, name in data_util_dict.items():

            if data_name in name:
                data_util_obj.get_data(data_name,
                                       train_splits,
                                       val_splits,
                                       shuffle_flag,
                                       random_seed)
                return data_util_obj

        print("current version support default datasets:")
        for data_util_obj, name in data_util_dict.items():

            for default_data_name in name:

                print(default_data_name)

        raise Exception("sorry current version don't support this default datasets", data_name)


if __name__ == "__main__":
    graph = Planetoid("cora",
                      train_splits=0.85,
                      val_splits=0.05,
                      shuffle_flag=False,
                      random_seed=None).data
    pass
