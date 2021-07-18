import requests
import unittest
from web_service.flask_app.codebase.prediction_service import ToneClassifier


class test_tone_classifier(unittest.TestCase):

    def setUp(self):
        self.URL = "http://127.0.0.1/predict"

        self.training_data_dict={1:'data/ma_data/ma1_FV1_MP3.mp3',
                                 2:'data/ma_data/ma2_FV1_MP3.mp3',
                                 3:'data/ma_data/ma3_FV1_MP3.mp3',
                                 4:'data/ma_data/ma4_FV1_MP3.mp3'}


    def test_training_data(self):

        for tone,path in self.training_data_dict.items():
            audio_file = open(path, "rb")
            values = {"audio_data" :(path,audio_file,"audio/mp3")}
            response = requests.post(self.URL, files=values,verify=True)
            data = response.json()
            assert data['tone']==tone

