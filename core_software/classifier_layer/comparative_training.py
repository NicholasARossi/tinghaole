import os
import numpy as np
import pandas as pd
from core_software.classifier_layer.models import cnnv1,lstmv1,mixtv1
from core_software.classifier_layer.visualizations import performance_plots_multi_model,plot_multiple_confusion_matrixes
from sklearn.model_selection import StratifiedShuffleSplit
import logging
from core_software.utils.config_params import phoneme_categories
import time
import pickle
import argparse
from pathlib import Path

dirname = os.path.abspath(os.path.dirname(__file__))


class Gauntlet:


    model_type_lookup = {'cnnv1':cnnv1,
                         'lstmv1':lstmv1,
                         'mixtv1':mixtv1}

    def __init__(self,
                 classifier_type:str='tone',
                 data_encoding:str='MSG',
                 model_types:list=['lstmv1'],
                 cv_folds:int=5,
                 n_epochs:int=2,
                 out_path:str='data/output/',
                 save_models:bool=True)-> None:

        self.classifier_type = classifier_type
        self.model_types=model_types
        self.cv_folds=cv_folds
        self.n_epochs=n_epochs
        self.data_encoding=data_encoding
        self.out_path=out_path
        self.save_models=save_models

        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)


    def _load_data(self) -> None:


        encoding_lookup={'MSG':os.path.join(dirname,'../../data/all_data/MSGs.npy'),
                         'MFCC':os.path.join(dirname,'../../data/all_data/mfccs.npy')}

        if self.data_encoding in encoding_lookup:
            encoded_data = np.asarray(np.load(encoding_lookup[self.data_encoding]).tolist())
            dim_1 = encoded_data.shape[1]
            dim_2 = encoded_data.shape[2]
            encoded_data = encoded_data.reshape((encoded_data.shape[0], dim_2, dim_1))
            self.X= np.asarray(encoded_data)



        else:
            logging.error(f'{self.data_encoding} not amoung encodings')




        full_labels = np.load(os.path.join(dirname,'../../data/all_data/full_labels.npy'))

        classifier_type_lookup={'tone':self._convert_tone_labels,
                         'phoneme':self._convert_phoneme_labels}

        if self.classifier_type in classifier_type_lookup:
            self.y=classifier_type_lookup[self.classifier_type](full_labels)
            self.num_classes=self.y.shape[1]


        else:
            logging.error(f'{self.classifier_type} not amoung encodings')



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
            split = StratifiedShuffleSplit(n_splits=self.cv_folds, test_size=0.16, random_state=88)
            split.get_n_splits(model_class.featurized_X, self.y)

            fold_count=0
            for idx1, idx2 in split.split(model_class.featurized_X, model_class.y):
                start = time.time()

                print(f'Starting fold: {fold_count+1}/{self.cv_folds}')
                model = model_class.get_model()
                X_train, X_test = model_class.featurized_X[idx1], model_class.featurized_X[idx2]
                y_train, y_test = model_class.y[idx1], model_class.y[idx2]

                history = model.fit(X_train, y_train, epochs=self.n_epochs, verbose=0, validation_data=(X_test, y_test))
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
        self._load_data()
        self._train_models()
        self._plotting()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--classifier_type', dest='classifier_type',
                        default='tone', type=str,
                        help='type of classification')

    parser.add_argument('-d', '--data_encoding', dest='data_encoding',
                        default='MSG', type=str,
                        help='data encoding')

    parser.add_argument('-m', '--model_types', dest='model_types', nargs='+', type=str,
                        help="type each model as a separate string separate by space")

    parser.add_argument('-v', '--cv_folds', dest='cv_folds',
                        default=5, type=int,
                        help='Number of CV folds')

    parser.add_argument('-n', '--n_epochs', dest='n_epochs',
                        default=5, type=int,
                        help='Number of epochs')

    parser.add_argument('-o', '--out_path', dest='out_path',
                        default='../../data/output/', type=str,
                        help='output path')


    args = parser.parse_args()

    tournament = Gauntlet(
        classifier_type=args.classifier_type,
        data_encoding=args.data_encoding,
        model_types=args.model_types,
        out_path=args.out_path,
        cv_folds=args.cv_folds,
        n_epochs=args.n_epochs
    )
    tournament.run()