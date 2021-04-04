
import numpy as np
from core_software.classifier_layer.models import get_cnn_model
from core_software.classifier_layer.visualizations import performance_plots,plot_confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

mfccs = np.load('../../data/all_data/MSGs.npy').tolist()
mfccs = np.asarray(mfccs)
labels = np.load('../../data/all_data/labels.npy')

dim_1 = mfccs.shape[1]
dim_2 = mfccs.shape[2]
channels = 1
classes = 4

X = mfccs
print(X.shape)
X = X.reshape((mfccs.shape[0], dim_1, dim_2, channels))
print(X.shape)
y = labels

onehot_y = np.zeros((y.size, y.max()))
onehot_y[np.arange(y.size),y-1] = 1

input_shape = (dim_1, dim_2, channels)
model = get_cnn_model(input_shape, classes)

#
# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.16, random_state=9)
split.get_n_splits(X, onehot_y)

for idx1, idx2 in split.split(X, y):
    X_train, X_test = X[idx1], X[idx2]
    y_train, y_test = onehot_y[idx1], onehot_y[idx2]

history = model.fit(X_train, y_train, epochs=15, verbose=1, validation_data=(X_test, y_test))

performance_plots(history,'CNN trained MSG','../../notebooks/figures/full_performance_cnn.png')

y_pred = model.predict(X_test).ravel()
y_pred_labels=model.predict_classes(X_test)
y_true_labels = np.argmax(y_test, axis=1)

plot_confusion_matrix(y_true_labels, y_pred_labels,
                          np.arange(classes)+1,
                          normalize=False,
                          title='Confusion matrix',
                          out_target='../../notebooks/figures/full_confusion.png')