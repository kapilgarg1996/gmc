import unittest
import os
import numpy as np
from gmc.conf import settings
from gmc.test import TestCase
from gmc.dataset import musicset

class TestEncodingDummy(unittest.TestCase):
    def setUp(self):
        self.dataset = musicset.MusicSet()
        self.dataset.one_hot_encode_genres()

    def test_genre_encoding(self):
        encoded_vectors = []
        total_genres = len(self.dataset.genres)
        for genre in self.dataset.genres:
            encoded_vectors.append(self.dataset.encoded_genres[genre])

        encoded_matrix = np.vstack(encoded_vectors)
        non_zero = encoded_matrix.nonzero()

        self.assertEqual(len(non_zero[0]), total_genres)
        self.assertEqual(len(non_zero[1]), total_genres)
        self.assertListEqual(non_zero[0].tolist(), non_zero[1].tolist())