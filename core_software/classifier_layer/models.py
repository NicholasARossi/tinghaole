from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,MaxPooling1D, BatchNormalization,LSTM,Input,Conv1D
from keras.models import Sequential
import keras




def get_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape))
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
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def get_LSTM_model(input_shape, num_classes):
    features=input_shape[1]
    time_steps=input_shape[0]

    look_back=32

    model = Sequential()

    model.add(LSTM(look_back, input_shape=(time_steps,features)))
    model.add(Dense(128, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def get_LSTM_model(input_shape, num_classes):
    n_features = input_shape[1]
    time_steps = input_shape[0]

    model = Sequential()

    model.add(Input(shape=(time_steps, n_features)))
    model.add(Conv1D(filters=32,
                     kernel_size=3,
                     strides=1,
                     activation='relu',
                     padding='same'))
    model.add(MaxPooling1D(data_format='channels_first', pool_size=(4)))
    model.add(LSTM(8))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model