from glob import glob
from keras.models import load_model
import numpy as np
import librosa
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def trim_leading_trailing_silence(audio):
    trimmed_sound = librosa.effects.trim(audio, top_db=20, frame_length=256, hop_length=64)[0]
    return trimmed_sound

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


class ToneClassifier:
    '''
    A determanistic tone ensemble model
    '''
    mapping_dict={0:1,
                  1:2,
                  2:3,
                  3:4}

    def __init__(self,
                 tone_dir:str ='./model_storage/tone/'):


        ### instantiate models
        script_dir = os.path.dirname(__file__)
        self.tone_models=self._instantiate_models( os.path.join(script_dir, tone_dir))


    def _instantiate_models(self,model_dir) -> list:
        models=[]

        model_targets=glob(model_dir+'/*.h5')

        if len(model_targets)!=5:
                logging.error(f'Incorrect number of models loaded {len(model_targets)}')

        for model_loc in model_targets:
            model = load_model(model_loc)
            models.append(model)
        return models


    def predict(self,audio_file):


        featurized_audio=self.featurize(audio_file)

        # make ensemble predictions from model
        predictions=[]
        for model in self.tone_models:
            prediction=model.predict_step(featurized_audio)
            predictions.append(prediction)

        predicted_idx=int(np.argmax(np.mean(np.asarray(predictions), axis=0)))


        return self.mapping_dict[predicted_idx]


    @staticmethod
    def featurize(audio_file):
        msg_audio_file=mp3toMSG(audio_file)
        max_dims=(80, 60)
        fully_featurized = add_padding([msg_audio_file], bonus_padding=2,maxes=max_dims)[0]

        fully_featurized=np.expand_dims(fully_featurized, 0)

        dim_0,dim_1,dim_2 = fully_featurized.shape

        fully_featurized= fully_featurized.reshape((dim_0, dim_1,dim_2 ))

        return fully_featurized


if __name__ == '__main__':

    # TODO through warnings if : no sound, no matches, genrally bad
    #test_file='../../data/ma_data/ma1_FV1_MP3.mp3'
    test_file='../../../data/ma_data/ma1_USER2.wav'

    classifier=ToneClassifier()
    prediction=classifier.predict(test_file)
    print(prediction)
