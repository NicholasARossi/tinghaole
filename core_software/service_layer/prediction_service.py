from glob import glob
from keras.models import load_model
from core_software.classifier_layer.prep_training_data import mp3toMSG,add_padding
import numpy as np
from pydub import AudioSegment
import scipy





class ToneClassifier:
    '''
    A determanistic tone ensemble model
    '''
    mapping_dict={0:1,
                  1:2,
                  2:3,
                  3:4}

    def __init__(self,
                 tone_dir:str = '/Users/nicholas.rossi/Documents/Personal/tinghaole/core_software/service_layer/model_storage/tone/'):


        ### instantiate models
        self.tone_models=self._instantiate_models(tone_dir)


    def _instantiate_models(self,model_dir) -> list:
        models=[]
        for model_loc in glob(model_dir+'*.h5'):
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
    test_file='../../data/ma_data/ma1_USER2.wav'

    classifier=ToneClassifier()
    prediction=classifier.predict(test_file)
    print(prediction)
