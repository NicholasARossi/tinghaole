import librosa
import numpy as np
import os
from glob import glob
import warnings
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
from tqdm import tqdm
import random
import tensorflow_io as tfio



class AugmentedDataGenerator:

    def __init__(self,data_dir,out_target='../../data/augmented_data',augment_multiplier=10,chunk_size=1000):
        self.data_dir=data_dir
        self.augment_multiplyer=augment_multiplier
        self.out_target=out_target
        self.chunk_size=chunk_size
        self.data_max_dims = (80, 60)
    def _load_raw_files(self):

        files = glob(f'{self.data_dir}*.mp3')
        ## suffle order of files
        random.shuffle(files)
        self.files=files

        ##

    def _iterate_file_list(self):



        ### loading original dataset
        MSGs = []
        full_labels = []

        for file in tqdm(self.files):
            failed=True
            while failed==True:
                ### hacky fix, data augmentation has a random chance to fail
                try:
                    MSG=mp3toMSG(file,trimming=True)
                    augmented=[]
                    for n in range(self.augment_multiplyer):
                        masked = tfio.audio.time_mask(MSG, param=10).numpy()
                        masked = tfio.audio.freq_mask(masked, param=10).numpy()
                        augmented.append(masked)
                    failed=False
                except:
                    print(f'{file} failed')
            MSGs.extend(augmented)
            full_labels.extend([file.split('/')[-1].replace('_MP3.mp3', '')]*len(augmented))

        # zip shuffle all data
        temp = list(zip(MSGs, full_labels))
        random.shuffle(temp)
        self.MSGs, self.full_labels = zip(*temp)



    def _serialize_data(self):


        MSG_dir=os.path.join(self.out_target,'MSG_chunks')
        Label_dir=os.path.join(self.out_target,'Label_chunks')
        for folder in [MSG_dir,Label_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        MSG_chunks = list(self.chunker(self.MSGs, self.chunk_size))
        Label_chunks= list(self.chunker(self.full_labels, self.chunk_size))

        chunk_number=0
        for MSG_chunk,Label_chunk in tqdm(zip(MSG_chunks,Label_chunks)):

            padded_data = add_padding(MSG_chunk, bonus_padding=2, maxes=self.data_max_dims)


            np.save(os.path.join(MSG_dir,f'chunk_{chunk_number:02d}.npy'), padded_data)
            np.save(os.path.join(Label_dir,f'chunk_{chunk_number:02d}.npy'), Label_chunk)

            chunk_number+=1

    def run_agumentation(self):
        self._load_raw_files()
        self._iterate_file_list()
        self._serialize_data()


    @staticmethod
    def chunker(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]


def trim_leading_trailing_silence(audio):
    trimmed_sound = librosa.effects.trim(audio, top_db=20, frame_length=256, hop_length=64)[0]
    return trimmed_sound

# prepping training data
def mp3tomfcc(file_path):
    audio, sample_rate = librosa.core.load(file_path)




    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=60)
    return mfcc


def mp3toMSG(file_path,trimming=True):
    audio, sample_rate = librosa.core.load(file_path)

    if trimming==True:
        audio=trim_leading_trailing_silence(audio)

    MSG = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=1024, hop_length=512, n_mels=80, fmin=75,
                                         fmax=3700)
    MSG = np.log10(MSG + 1e-10)

    return MSG


def add_padding(arrays, bonus_padding=10,maxes=None):
    if maxes is None:
        heights, widths = list(zip(*[x.shape for x in arrays]))
        max_height = max(heights)
        max_width = max(widths)

    else:
        max_height,max_width=maxes[0],maxes[1]

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






if __name__ == '__main__':
    augmentation_engine=AugmentedDataGenerator('../../data/tone_perfect_mp3/')
    augmentation_engine.run_agumentation()