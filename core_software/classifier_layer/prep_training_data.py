import librosa
import numpy as np
import os
from glob import glob
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
from tqdm import tqdm

# prepping training data
def mp3tomfcc(file_path, max_pad):
    audio, sample_rate = librosa.core.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    return mfcc


def mp3toMSG(file_path, max_pad):
    audio, sample_rate = librosa.core.load(file_path)
    MSG = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=80, fmin=75,
                                         fmax=3700)
    MSG = np.log10(MSG + 1e-10)

    return MSG


def add_padding(arrays, bonus_padding=10):
    heights, widths = list(zip(*[x.shape for x in arrays]))
    max_height = max(heights)
    max_width = max(widths)

    new_arrays = []
    for array in arrays:
        pad_width = (max_width - array.shape[1]) + bonus_padding
        pad_height = (max_height - array.shape[0]) + bonus_padding

        new_array = np.pad(array, pad_width=((bonus_padding, pad_height), (bonus_padding, pad_width)), mode='constant')
        new_arrays.append(new_array)

    heights, widths = list(zip(*[x.shape for x in new_arrays]))

    assert len(set(heights)) == 1, print('not all the same size!')
    assert len(set(widths)) == 1, print('not all the same size!')
    return new_arrays



### PREPPING MA DATA

files = glob('../../data/tone_perfect_mp3/ma*.mp3')
substring_list = ['ma1', 'ma2', 'ma3', 'ma4']
mfccs = []
MSGs = []
labels = []
full_labels = []

for file in files:
    if any(map(file.__contains__, substring_list)):
        mfccs.append(mp3tomfcc(file, 60))
        MSGs.append(mp3toMSG(file, 60))

        label = file.split('/')[-1].split('_')[0][-1]
        labels.append(int(label))

        # full labels
        full_labels.append(file.split('/')[-1].replace('_MP3.mp3', ''))

MSGs = add_padding(MSGs, bonus_padding=2)
mfccs = add_padding(mfccs, bonus_padding=2)

mfccs, MSGs, labels, full_labels = list(zip(*sorted(zip(mfccs, MSGs, labels, full_labels), key=lambda x: x[2])))

np.save('../../data/ma_data/ma_mfccs.npy', mfccs)
np.save('../../data/ma_data/ma_MSGs.npy', MSGs)
np.save('../../data/ma_data/ma_labels.npy', labels)

# ### PREPPING ALL DATA
#
files = glob('../../data/tone_perfect_mp3/*.mp3')

mfccs = []
MSGs = []
labels = []
full_labels = []

for file in tqdm(files):
    mfccs.append(mp3tomfcc(file, 60))
    MSGs.append(mp3toMSG(file, 60))

    label = file.split('/')[-1].split('_')[0][-1]
    labels.append(int(label))

    # full labels
    full_labels.append(file.split('/')[-1].replace('_MP3.mp3', ''))

MSGs = add_padding(MSGs, bonus_padding=2)
mfccs = add_padding(mfccs, bonus_padding=2)

mfccs, MSGs, labels, full_labels = list(zip(*sorted(zip(mfccs, MSGs, labels, full_labels), key=lambda x: x[2])))

np.save('../../data/all_data/mfccs.npy', mfccs)
np.save('../../data/all_data/MSGs.npy', MSGs)
np.save('../../data/all_data/labels.npy', labels)
np.save('../../data/all_data/full_labels.npy', full_labels)


