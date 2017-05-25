import unittest
import os
from gmc.conf import settings
from gmc.test import TestCase
from gmc.dataset import musicset, features

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestTempFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fileset = musicset.MusicSet()
        cls.fileset.load_files()

    def test_feature_computation(self):
        module_path = os.path.dirname(os.path.abspath(__file__))
        test_file = os.path.join(module_path, 'blues.00099.au.wav')
        try:
            mfcc_features = features.mfcc(test_file)
            dwt_features = features.dwt(test_file)
            beat_features = features.beat(test_file)
        except Exception as e:
            self.fail(e)

    @classmethod
    def tearDownClass(cls):
        cls.fileset.destroy_results()

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestRealFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fileset = musicset.MusicSet(dirname='features')
        cls.fileset.load_files()

    def test_train_data_loading(self):
        train = self.fileset.load_train_data()
        self.assertEqual(train.music.shape[0], 700)
        self.assertEqual(train.music.shape[1], 188)
        self.assertEqual(train.labels.shape[0], 700)
        self.assertEqual(train.labels.shape[1], 10)

    def test_test_data_loading(self):
        test = self.fileset.load_test_data()
        self.assertEqual(test.music.shape[0], 300)
        self.assertEqual(test.music.shape[1], 188)
        self.assertEqual(test.labels.shape[0], 300)
        self.assertEqual(test.labels.shape[1], 10)