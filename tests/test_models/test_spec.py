import unittest
import os
import numpy as np
from gmc.core.models import specfeatures as sf
from gmc.dataset import musicset, reduce
from gmc.conf import settings

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestNN(unittest.TestCase):
    def test_training(self):
        p = os.path.join(settings.BRAIN_DIR, "specfeat")
        if not os.path.isdir(p):
            os.mkdir(p)
        dataset = musicset.MusicSet(dirname='features_s_10', force_load=True)
        dataset.load_files()
        dataset.one_hot_encode_genres()
        cnn_t = sf.CNNFeat(dataset)
        ds = cnn_t.train(display_step=100, path=p)
        print(ds.music.shape)
        print(ds.labels.shape)