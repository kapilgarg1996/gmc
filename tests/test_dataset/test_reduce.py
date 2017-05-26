import unittest
import os
from gmc.conf import settings
from gmc.core.cache import store
from gmc.dataset import reduce, musicset

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestPCA(unittest.TestCase):
    def test_pca_mdb(self):
        featureset = musicset.MusicSet(dirname='features2')
        train = featureset.load_train_data()
        new_features = reduce.pca(train.music)
        storage = store(featureset.results_dir)
        storage['reduced_mdb.dat'] = new_features
        self.assertGreater(train.music.shape[1], new_features.shape[1])

    def test_lda(self):
        featureset = musicset.MusicSet(dirname='features2')
        train = featureset.load_train_data()
        new_features = reduce.lda(train.music, train.labels)
        print(new_features.shape)

    def test_visual_data(self):
        featureset = musicset.MusicSet(dirname='features2')
        featureset.visualize()