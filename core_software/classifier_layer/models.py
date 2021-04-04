from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,MaxPooling1D, BatchNormalization,LSTM,Input,Conv1D
from keras.models import Sequential
import keras




class cnnv1:
    description='A basic CNN model'

    def __init__(self, X, y, num_classes):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self._featurize()

    def _featurize(self):
        dim_0 = self.X.shape[0]
        dim_1 = self.X.shape[1]
        dim_2 = self.X.shape[2]
        channels = 1
        self.featurized_X= self.X.reshape((dim_0, dim_1, dim_2, channels))
        self.input_shape= (dim_1, dim_2, channels)


    def get_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(48, kernel_size=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(120, kernel_size=(2, 2), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

class lstmv1:
    description='A basic LSTM model'

    def __init__(self, X, y, num_classes):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self._featurize()

    def _featurize(self):
        dim_0 = self.X.shape[0]
        dim_1 = self.X.shape[1]
        dim_2 = self.X.shape[2]
        self.featurized_X= self.X.reshape((dim_0, dim_2,dim_1 ))
        self.input_shape= (dim_2,dim_1)

    def get_model(self):
        time_steps=self.input_shape[0]
        features=self.input_shape[1]

        look_back=32

        model = Sequential()

        model.add(LSTM(look_back, input_shape=(time_steps,features)))
        model.add(Dense(128, activation='relu'))

        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class mixtv1:
    description='A basic CNN/LSTM hybrid model'

    def __init__(self, X, y, num_classes):
        self.X = X
        self.y = y
        self.num_classes = num_classes
        self._featurize()

    def _featurize(self):
        dim_0 = self.X.shape[0]
        dim_1 = self.X.shape[1]
        dim_2 = self.X.shape[2]
        self.featurized_X= self.X.reshape((dim_0, dim_2,dim_1 ))
        self.input_shape= (dim_2,dim_1)



    def get_model(self):
        time_steps=self.input_shape[0]
        features=self.input_shape[1]

        model = Sequential()

        model.add(Input(shape=(time_steps, features)))
        model.add(Conv1D(filters=32,
                         kernel_size=3,
                         strides=1,
                         activation='relu',
                         padding='same'))
        model.add(MaxPooling1D(data_format='channels_first', pool_size=(4)))
        model.add(LSTM(8))
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='adam',
                      metrics=['accuracy'])
        return model
