import unittest
import os
from gmc.conf import settings
from gmc.test import TestCase
from gmc.dataset import musicset

class TestDummyFiles(TestCase):
    def setUp(self):
        self.fileset = musicset.MusicSet()
        self.fileset.load_files()

    def test_genres_count(self):
        self.assertEqual(len(self.fileset.files), 10)

    def test_files_count_per_genre(self):
        for genre in self.fileset.files:
            self.assertEqual(len(self.fileset.files[genre]), 5)

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestReadFiles(unittest.TestCase):
    def setUp(self):
        self.fileset = musicset.MusicSet()
        self.fileset.load_files()

    def test_genres_count(self):
        self.assertEqual(len(self.fileset.files), 10)