from keras.utils.data_utils import Sequence
import pandas as pd
import numpy as np
import os


class DataGenerator(Sequence):

    def __init__(self, file_list, class_list, data_dir):
        """Constructor can be expanded,
           with batch size, dimentation etc.
        """
        self.file_list = file_list
        self.class_list = class_list
        self.data_dir = data_dir
        self.on_epoch_end()

    def __len__(self):
        'Take all batches in each iteration'
        return int(len(self.file_list))

    def __getitem__(self, index):
        'Get next batch'
        # Generate indexes of the batch
        indexes = self.indexes[index:(index + 1)]

        # single file
        file_list_temp = [self.file_list[k] for k in indexes]
        class_list_temp = [self.class_list[k] for k in indexes]
        # Set of X_train and y_train
        X, y = self.__data_generation(file_list_temp, class_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.file_list))

    def __data_generation(self, file_list_temp, class_list_temp):
        'Generates data containing batch_size samples'
        # Generate data
        arrays = []
        for ID in file_list_temp:
            x_file_path = os.path.join(self.data_dir, ID)

            # Store sample
            arrays.append(np.load(x_file_path))

        y = np.asarray(class_list_temp)

        onehot_y = np.zeros((len(class_list_temp), 4))
        onehot_y[np.arange(y.size), y - 1] = 1
        return np.expand_dims(np.asarray(arrays), -1), onehot_y
