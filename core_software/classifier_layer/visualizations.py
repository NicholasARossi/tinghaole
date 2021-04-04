from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
import pickle
import seaborn as sns
import pandas as pd

plt.style.use('rossidata')

def return_mean_std(arrayo):
    stds = np.std(arrayo, axis=0)
    means = np.mean(arrayo, axis=0)
    epoch_array = np.arange(len(stds))
    lower_bound = means - stds
    upper_bound = means + stds
    return means,lower_bound,upper_bound,epoch_array


def performance_plots_multi_model(serialized_histories, out_target='figures/cnn_performance.png'):
    colors = sns.color_palette("Set2", len(serialized_histories))

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    for i,serialized_history in enumerate(serialized_histories):
        with open(serialized_history, 'rb') as handle:
            unserialized_history = pickle.load(handle)

        name=serialized_history.split('/')[-1].replace('.pkl','')

        metrics=['loss_array','accuracy_array','val_loss_array','val_accuracy_array']

        for j,axo in enumerate(ax.reshape(-1)):

            means,lower_bound,upper_bound,epoch_array=return_mean_std(unserialized_history[metrics[j]])

            axo.fill_between(epoch_array, lower_bound, upper_bound, facecolor=colors[i],alpha=0.1)
            axo.plot(epoch_array,means, color=colors[i],linewidth=3,label=name)
            axo.set_title(f'{metrics[j]}')
            axo.set_xlabel('Epochs')

    axo.legend()

    fig.savefig(out_target, dpi=300, bbox_inches='tight')

def performance_plots(history, label='CNN', out_target='figures/cnn_performance.png'):
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    ax[0, 0].plot(history.history['loss'])
    ax[0, 0].set_title(f'{label} loss')
    ax[0, 0].set_xlabel('Epochs')

    ax[0, 1].plot(history.history['accuracy'])
    ax[0, 1].set_title(f'{label} accuracy')
    ax[0, 1].set_xlabel('Epochs')

    ax[1, 0].plot(history.history['val_loss'])
    ax[1, 0].set_title(f'{label} val_loss')
    ax[1, 0].set_xlabel('Epochs')

    ax[1, 1].plot(history.history['val_accuracy'])
    ax[1, 1].set_title(f'{label} val_accuracy')
    ax[1, 1].set_xlabel('Epochs')

    fig.savefig(out_target, dpi=300, bbox_inches='tight')


def plot_multiple_confusion_matrixes(serialized_histories,labels,out_target):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    conter=len(serialized_histories)
    fig, ax = plt.subplots(1, conter, figsize=(4*conter,4))



    for i, serialized_history in enumerate(serialized_histories):
        with open(serialized_history, 'rb') as handle:
            unserialized_history = pickle.load(handle)
        name=serialized_history.split('/')[-1].replace('.pkl','')

        cm = confusion_matrix(unserialized_history['true_values'], unserialized_history['predictions'])
        cmn = cm.astype('float') /cm.sum(axis=1)[:, np.newaxis]

        df_cm = pd.DataFrame(cmn, index=labels,
                             columns=labels)
        sns.heatmap(df_cm, annot=True,ax=ax[i],cmap='Spectral',linewidths=.5)
        ax[i].set_title(name)

    fig.savefig(out_target, dpi=300, bbox_inches='tight')
