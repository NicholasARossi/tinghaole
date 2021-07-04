# train the model
# for e in range(total_epoch):
#     print("Epoch =", e + 1, "out of", total_epoch)
#     for f in range(num_chunks - 1):
#         with gzip.GzipFile(self.feature_chunks[f], "r") as xhandle:
#             X_train = np.load(xhandle)
#
#         y_train = np.load(self.label_chunks[f])
#
#         history = model.fit(X_train, y_train, batch_size=8, \
#                             validation_split=0.0, epochs=1, verbose=1, class_weight=self.class_weights)
#         epoch_train_acc[e, f] = history.history['accuracy'][0]
#
#     # train final chunk and do validation
#     with gzip.GzipFile(self.feature_chunks[num_chunks], "r") as xhandle:
#
#         X_train = np.load(xhandle)
#
#     y_train = np.load(self.label_chunks[num_chunks])
#
#     history = model.fit(X_train, y_train, batch_size=8, \
#                         validation_data=(self.val_seqs, self.val_labels), epochs=1, verbose=1,
#                         class_weight=self.class_weights)
#     epoch_train_acc[e, num_chunks - 1] = history.history['accuracy'][0]
#     epoch_val_acc[e, 0] = history.history['val_acc'][0]
#
#     # record max validation accuracy
#     if history.history['val_acc'][0] > max_val_acc:
#         max_val_acc = history.history['val_acc'][0]
#         max_acc_pair = history.history['accuracy'][0]


import os
import numpy as np
import pandas as pd
from core_software.classifier_layer.models import cnnv1,lstmv1,mixtv1
from core_software.classifier_layer.visualizations import performance_plots_multi_model,plot_multiple_confusion_matrixes
from sklearn.model_selection import KFold
import logging
from core_software.utils.config_params import phoneme_categories
import pickle
import time
from glob import glob
from pathlib import Path
import keras

dirname = os.path.abspath(os.path.dirname(__file__))


class Gauntlet:


    model_type_lookup = {'cnnv1':cnnv1,
                         'lstmv1':lstmv1,
                         'mixtv1':mixtv1}

    def __init__(self,
                 classifier_type:str='tone',
                 model_types:list=['lstmv1','mixtv1','cnnv1'],
                 cv_folds:int=5,
                 n_epochs:int=2,
                 out_path:str='data/output/',
                 save_models:bool=True,
                 data_dir:str='../../data/augmented_data')-> None:

        self.classifier_type = classifier_type
        self.model_types=model_types
        self.cv_folds=cv_folds
        self.n_epochs=n_epochs
        self.out_path=out_path
        self.save_models=save_models
        self.data_dir=data_dir

        self.MSG_dir=os.path.join(self.data_dir,'MSG_chunks')
        self.Label_dir=os.path.join(self.data_dir,'Label_chunks')

        self.n_data_chunks=len(glob(self.MSG_dir+'/*.npy'))

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

        # perform taste test
        self.X=self._load_and_reshape(self.MSG_dir, 0)

        if self.classifier_type == 'tone':
            self.num_classes = 4

        elif self.classifier_type == 'phoneme':
            self.num_classes = len(phoneme_categories)

        else:
            logging.error('Classifier Type not detected')

    @staticmethod
    def _load_and_reshape(folder,number):
        array_target = os.path.join(folder, f'chunk_{number:02d}.npy')
        encoded_data = np.asarray(np.load(array_target).tolist())

        dim_1 = encoded_data.shape[1]
        dim_2 = encoded_data.shape[2]
        encoded_data = encoded_data.reshape((encoded_data.shape[0], dim_2, dim_1))
        return encoded_data

    def _load_data(self,chunk_num) -> [np.ndarray,list,int]:

        # load the MSGs for this particular chunk
        encoded_data = self._load_and_reshape(self.MSG_dir,chunk_num)

        # load labels and convert
        label_target=os.path.join(self.Label_dir, f'chunk_{chunk_num:02d}.npy')

        full_labels = np.load(label_target)

        classifier_type_lookup={'tone':self._convert_tone_labels,
                         'phoneme':self._convert_phoneme_labels}


        encoded_labels=classifier_type_lookup[self.classifier_type](full_labels)

        return encoded_data,encoded_labels


    def _load_assemble_test_data(self,chunk_num_list):


        # assemble all chunks from the test data into one
        data=[]
        labels=[]
        for z in chunk_num_list:
            array_target = os.path.join(self.MSG_dir, f'chunk_{z:02d}.npy')
            encoded_data = np.asarray(np.load(array_target).tolist())

            dim_1 = encoded_data.shape[1]
            dim_2 = encoded_data.shape[2]
            encoded_data = encoded_data.reshape((encoded_data.shape[0], dim_2, dim_1))
            data.append(encoded_data)

            label_target = os.path.join(self.Label_dir, f'chunk_{z:02d}.npy')

            full_labels = np.load(label_target)
            labels.append(full_labels)
        return data, labels

    @staticmethod
    def _convert_tone_labels(labels:np.ndarray)-> np.ndarray:

        y=np.asarray([int(label.split('_')[0][-1]) for label in labels])
        onehot_y = np.zeros((y.size, y.max()))
        onehot_y[np.arange(y.size), y - 1] = 1

        return onehot_y

    @staticmethod
    def _convert_phoneme_labels(labels:np.ndarray)-> np.ndarray:
        string_phonemes=[label.split('_')[0][:-1] for label in labels]
        y=np.asarray([phoneme_categories.index(label) for label in string_phonemes])
        onehot_y = np.zeros((y.size, y.max()))
        onehot_y[np.arange(y.size), y - 1] = 1
        return onehot_y


    def _train_models(self):


        for model_type in self.model_types:
            print(f'Starting {model_type}')
            histories = {}
            model_class=self.model_type_lookup[model_type](self.X,self.y,self.num_classes)


            split = KFold(n_splits=self.cv_folds)

            split.get_n_splits(model_class.featurized_X, self.y)

            fold_count=0
            for train_idx, test_idx in split.split(np.arange(self.n_data_chunks)):
                start = time.time()

                print(f'Starting fold: {fold_count+1}/{self.cv_folds}')
                model = model_class.get_model()



                history = model.fit_generator(X_train, y_train, epochs=self.n_epochs, verbose=0, validation_data=(X_test, y_test))
                y_pred=model.predict(X_test)

                history.history.update({'y_pred':y_pred,
                                 'y_pred_labels':np.argmax(y_pred, axis=1),
                                 'y_true_labels':np.argmax(y_test, axis=1),
                                        'time':time.time()-start})



                histories[f'{model_type}_{fold_count}']=history.history

                if self.save_models == True:
                    subdir_path=os.path.join(self.out_path,model_type)
                    Path(subdir_path).mkdir(parents=True, exist_ok=True)

                    model_save_loc = os.path.join(subdir_path,f'{model_type}_{fold_count}.h5')
                    model.save(model_save_loc)





                fold_count+=1

            self._serializing_outputs(histories,model_type)

    def _serializing_outputs(self,histories,model_label):

        loss_array=np.zeros((self.cv_folds,self.n_epochs))
        accuracy_array=np.zeros((self.cv_folds,self.n_epochs))

        val_loss_array=np.zeros((self.cv_folds,self.n_epochs))
        val_accuracy_array=np.zeros((self.cv_folds,self.n_epochs))
        time_array=np.zeros((self.cv_folds,1))
        all_labels_pred=[]
        all_labels_true=[]

        for i,key in enumerate(histories.keys()):
            loss_array[i,:]=histories[key]['loss']
            accuracy_array[i, :] = histories[key]['accuracy']
            val_loss_array[i, :] = histories[key]['val_loss']
            val_accuracy_array[i, :] = histories[key]['val_accuracy']
            time_array[i]=histories[key]['time']
            all_labels_pred.extend(histories[key]['y_pred_labels'])
            all_labels_true.extend(histories[key]['y_true_labels'])

        serialized_dict={}
        serialized_dict['loss_array']=loss_array
        serialized_dict['accuracy_array']=accuracy_array
        serialized_dict['val_loss_array']=val_loss_array
        serialized_dict['val_accuracy_array']=val_accuracy_array
        serialized_dict['times']=time_array
        serialized_dict['predictions']=all_labels_pred
        serialized_dict['true_values']=all_labels_true


        with open(self.out_path+f'{model_label}.pkl', 'wb') as handle:
            pickle.dump(serialized_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def _plotting(self):

        all_serialized_outputs=[self.out_path+f'{model_type}.pkl' for model_type in self.model_types]

        performance_plots_multi_model(all_serialized_outputs,out_target=self.out_path+f'performance_plots.png')


        cm_label_encodings={'tone':['1','2','3','4'],
                            'phoneme':phoneme_categories}
        plot_multiple_confusion_matrixes(all_serialized_outputs,cm_label_encodings[self.classifier_type],out_target=self.out_path+f'cm_plots.png')


    def run(self):
        self._train_models()


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, x_col, y_col=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.df = dataframe
        self.indices = self.df.index.tolist()
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.x_col = x_col
        self.y_col = y_col
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size)

        def __getitem__(self, index):
            index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
            batch = [self.indices[k] for k in index]

            X, y = self.__get_data(batch)
            return X, y

        def on_epoch_end(self):
            self.index = np.arange(len(self.indices))
            if self.shuffle == True:
                np.random.shuffle(self.index)

        def __get_data(self, batch):
            X =  # logic
            y =  # logic

            for i, id in enumerate(batch):
                X[i,] =  # logic
                y[i] =  # labels

            return X, y

if __name__ == '__main__':
    tournament = Gauntlet()
    tournament.run()

# if __name__ == "__main__":
#
#
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument('-c', '--classifier_type', dest='classifier_type',
#                         default='tone', type=str,
#                         help='type of classification')
#
#     parser.add_argument('-d', '--data_encoding', dest='data_encoding',
#                         default='MSG', type=str,
#                         help='data encoding')
#
#     parser.add_argument('-m', '--model_types', dest='model_types', nargs='+', type=str,
#                         help="type each model as a separate string separate by space")
#
#     parser.add_argument('-v', '--cv_folds', dest='cv_folds',
#                         default=5, type=int,
#                         help='Number of CV folds')
#
#     parser.add_argument('-n', '--n_epochs', dest='n_epochs',
#                         default=5, type=int,
#                         help='Number of epochs')
#
#     parser.add_argument('-o', '--out_path', dest='out_path',
#                         default='../../data/output/', type=str,
#                         help='output path')
#
#
#     args = parser.parse_args()
#
#     tournament = Gauntlet(
#         classifier_type=args.classifier_type,
#         data_encoding=args.data_encoding,
#         model_types=args.model_types,
#         out_path=args.out_path,
#         cv_folds=args.cv_folds,
#         n_epochs=args.n_epochs
#     )
#     tournament.run()