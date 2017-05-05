"""
Class MusicSet holds the files and their respective
genres
""" 
import os
import pickle
import shutil
import numpy as np
from gmc.conf import settings
from gmc.core.cache import store

RESULT_DIR = 'musicset'

class MusicSet:
    results_dir = os.path.join(settings.BRAIN_DIR, RESULT_DIR)
    force_load = False
    def __init__(self, force_load=False, genres=None):
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.force_load = force_load
        self.genres = genres or settings.GENRES
        self.files = {}
        self.train = None
        self.test = None
        self.encoded_genres = {}

    @store(os.path.join(results_dir, 'loaded_files.dat'), 'files', force=force_load)
    def load_files(self):
        for genre in self.genres:
            genre_path = os.path.join(settings.DATASET_DIR, genre)
            if os.path.isdir(genre_path):
                self.files[genre] = []
                for f in os.listdir(genre_path):
                    file_path = os.path.join(genre_path, f)
                    if os.path.isfile(file_path) and f.endswith(".wav"):
                        self.files[genre].append(file_path)
            else:
                print('No Directory found for genre: '+genre)

    def one_hot_encode_genres(self):
        total_genres = len(self.genres)
        genre_class=0
        for genre in self.genres:
            encoded = self.encoded_genres[genre] = np.zeros(total_genres)
            encoded[genre_class] = 1
            genre_class += 1

    def destroy_results(self):
        shutil.rmtree(self.results_dir)