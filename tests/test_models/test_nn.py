import unittest
import os
import numpy as np
from gmc.core.models import nn
from gmc.dataset import musicset, reduce

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestNN(unittest.TestCase):
    def test_training(self):
        dataset = musicset.MusicSet(dirname='features_m_10_fl100')
        dataset.load_files()
        dataset.load_train_data()
        dataset.load_test_data()
        nn_t = nn.NN(dataset)
        nn_t.train(display_step=100)