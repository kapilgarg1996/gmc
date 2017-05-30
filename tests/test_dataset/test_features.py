import unittest
import os
import librosa
import numpy as np
from librosa import display
import matplotlib.pyplot as plt
from gmc.conf import settings
from gmc.test import TestCase
from gmc.dataset import musicset, features

class TestTempFiles(unittest.TestCase):
    def test_feature_computation(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(module_path, 'blues.00099.au.wav')
        try:
            mfcc_features = features.mfcc(test_file)
            dwt_features = features.dwt(test_file)
            beat_features = features.beat(test_file)
        except Exception as e:
            self.fail(e)

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestRealFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fileset = musicset.MusicSet(dirname='features3')
        cls.fileset.load_files()
        cls.total_features = 0
        for f in settings.FEATURES:
            cls.total_features += settings.FEATURES_LENGTH[f]

    def test_train_data_loading(self):
        train = self.fileset.load_train_data()
        self.assertEqual(train.music.shape[0], 700)
        self.assertEqual(train.music.shape[1], self.total_features)
        self.assertEqual(train.labels.shape[0], 700)
        self.assertEqual(train.labels.shape[1], 10)

    def test_test_data_loading(self):
        test = self.fileset.load_test_data()
        self.assertEqual(test.music.shape[0], 300)
        self.assertEqual(test.music.shape[1], self.total_features)
        self.assertEqual(test.labels.shape[0], 300)
        self.assertEqual(test.labels.shape[1], 10)

class TestPlots(unittest.TestCase):
    def test_mfcc_plot(self):
        plt.figure(figsize=(10, 4))
        module_path = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(module_path, 'blues.00099.au.wav')
        y, sr = librosa.load(test_file)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', 
            fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.show()