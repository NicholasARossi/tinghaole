import requests
import unittest
from web_service.flask_app.codebase.prediction_service import ToneClassifier


class test_tone_classifier(unittest.TestCase):

    def setUp(self):
        self.audio_path='../../data/ma_data/ma1_USER2.wav'

        self.training_data_dict={1:'data/ma_data/ma1_FV1_MP3.mp3',
                                 2:'data/ma_data/ma2_FV1_MP3.mp3',
                                 3:'data/ma_data/ma3_FV1_MP3.mp3',
                                 4:'data/ma_data/ma4_FV1_MP3.mp3'}

        self.classifier=ToneClassifier()


    def test_training_data(self):

        for tone,test_file in self.training_data_dict.items():
            prediction = self.classifier.predict(test_file)
            assert tone==prediction

