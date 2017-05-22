"""
Class MusicSet holds the files and their respective
genres
""" 
import os
import pickle
import shutil
import numpy as np
import importlib
from gmc.conf import settings
from gmc.core.cache import store
from gmc.dataset import features

RESULT_DIR = 'musicset'

class DataSet:
    music = None
    labels = None

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
        self.encoded_genres = None

    @store(os.path.join(results_dir, 'loaded_files.dat'), prop='files', force=force_load)
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
        self.encoded_genres = {}
        total_genres = len(self.genres)
        genre_class=0
        for genre in self.genres:
            encoded = self.encoded_genres[genre] = np.zeros(total_genres)
            encoded[genre_class] = 1
            genre_class += 1

    @store(os.path.join(results_dir, 'train.dat'), prop='train', force=force_load)
    def load_train_data(self):
        if self.train is None:
            self.train = DataSet()
        if self.encoded_genres is None:
            self.one_hot_encode_genres()
        tr_ratio = int(settings.TRAIN_TEST_RATIO[0])/np.sum(settings.TRAIN_TEST_RATIO)
        for genre in self.genres:
            num_total_files = len(self.files[genre])
            num_train_files = int(num_total_files*tr_ratio)
            train_files = self.files[genre][:num_train_files]
            for file in train_files:
                result = None
                for f in settings.FEATURES:
                    feat_func = getattr(features, f)
                    if result is None:
                        result = feat_func(file)
                    else:
                        result = np.append(result, feat_func(file))
                if self.train.music is None:
                    self.train.music = result
                    self.train.labels = np.array(self.encoded_genres[genre])
                else:
                    self.train.music = np.vstack((self.train.music, result,))
                    self.train.labels = np.vstack((self.train.labels, self.encoded_genres[genre]))


    @store(os.path.join(results_dir, 'test.dat'), prop='test', force=force_load)
    def load_test_data(self):
        if self.test is None:
            self.test = DataSet()
        if self.encoded_genres is None:
            self.one_hot_encode_genres()
        tr_ratio = int(settings.TRAIN_TEST_RATIO[0])/np.sum(settings.TRAIN_TEST_RATIO)
        for genre in self.genres:
            num_total_files = len(self.files[genre])
            num_train_files = int(num_total_files*tr_ratio)
            test_files = self.files[genre][num_train_files:]
            for file in test_files:
                result = None
                for f in settings.FEATURES:
                    feat_func = getattr(features, f)
                    if result is None:
                        result = feat_func(file)
                    else:
                        result = np.append(result, feat_func(file))
                if self.test.music is None:
                    self.test.music = result
                    self.test.labels = np.array(self.encoded_genres[genre])
                else:
                    self.test.music = np.vstack((self.test.music, result,))
                    self.test.labels = np.vstack((self.test.labels, self.encoded_genres[genre]))

    def destroy_results(self):
        shutil.rmtree(self.results_dir)