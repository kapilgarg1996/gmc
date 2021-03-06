"""
Class MusicSet holds the files and their respective
genres
""" 
import os
import pickle
import shutil
import numpy as np
import importlib
import random
from matplotlib import pyplot as plt
from gmc.conf import settings
from gmc.core.cache import store
from gmc.dataset import features, reduce

RESULT_DIR = 'musicset'

class DataSet:
    def __init__(self):
        self.music = None
        self.labels = None

    def next_batch(self, n):
        if self.music is None:
            return None, None

        idx = np.arange(self.music.shape[0])
        np.random.shuffle(idx)
        idx = idx[:n]
        return self.music[idx, :], self.labels[idx, :]

class MusicSet:
    def __init__(self, force_load=False, genres=None, dirname=RESULT_DIR):
        self.results_dir = os.path.join(settings.BRAIN_DIR, dirname)
        if not os.path.isdir(self.results_dir):
            os.mkdir(self.results_dir)
        self.genres = genres or settings.GENRES
        self.storage = store(self.results_dir, force_load)
        self.files = {}
        self.train = None
        self.test = None
        self.validation = None
        self.encoded_genres = None
        self.load_files()

    def load_files(self):
        self.files = self.storage['loaded_files.dat']
        if self.files is not None:
            return self.files
        self.files = {}
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
        self.storage['loaded_files.dat'] = self.files
        return self.files

    def one_hot_encode_genres(self):
        self.encoded_genres = {}
        total_genres = len(self.genres)
        genre_class=0
        for genre in self.genres:
            encoded = self.encoded_genres[genre] = np.zeros(total_genres)
            encoded[genre_class] = 1
            genre_class += 1

    def load_train_data(self):
        self.train = self.storage['train.dat']
        if self.train is not None:
            return self.train
        if self.train is None:
            self.train = DataSet()
        if self.encoded_genres is None:
            self.one_hot_encode_genres()
        tr_ratio = int(settings.TRAIN_TEST_RATIO[0])/np.sum(settings.TRAIN_TEST_RATIO)
        for genre in self.genres:
            print("Feautures for %s Extracting" % genre)
            num_total_files = len(self.files[genre])
            num_train_files = int(num_total_files*tr_ratio)
            train_files = self.files[genre][:num_train_files]
            for file in train_files:
                result = None
                for f in settings.FEATURES:
                    feat_func = getattr(features, f)
                    out = feat_func(file)
                    if result is None:
                        result = out
                    else:
                        result = np.append(result, out, axis=-1)
                if self.train.music is None:
                    self.train.music = result
                    n_samp = result.shape[0]
                    self.train.labels = np.tile(np.array(self.encoded_genres[genre]), (n_samp, 1))
                else:
                    self.train.music = np.vstack((self.train.music, result,))
                    n_samp = result.shape[0]
                    labels = np.tile(np.array(self.encoded_genres[genre]), (n_samp, 1))
                    self.train.labels = np.vstack((self.train.labels, labels))

        self.storage['train.dat'] = self.train
        return self.train


    def load_test_data(self):
        self.test = self.storage['test.dat']
        if self.test is not None:
            return self.test
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
                    out = feat_func(file)
                    if result is None:
                        result = out
                    else:
                        result = np.append(result, out, axis=-1)
                if self.test.music is None:
                    self.test.music = result
                    n_samp = result.shape[0]
                    self.test.labels = np.tile(np.array(self.encoded_genres[genre]), (n_samp, 1))
                else:
                    self.test.music = np.vstack((self.test.music, result,))
                    n_samp = result.shape[0]
                    labels = np.tile(np.array(self.encoded_genres[genre]), (n_samp, 1))
                    self.test.labels = np.vstack((self.test.labels, labels))

        self.storage['test.dat'] = self.test
        return self.test

    def load_validation_data(self, v_ratio=0.2):
        self.load_train_data()
        self.validation = DataSet()
        n_tr = self.train.music.shape[0]
        temp = None
        temp_l = None
        for x in range(0, n_tr, 100):
            if temp is None:
                temp = self.train.music[x:x+100]
                temp_l = self.train.labels[x:x+100]
            else:
                temp = np.hstack((temp, self.train.music[x:x+100]))
                temp_l = np.hstack((temp_l, self.train.labels[x:x+100]))

        self.validation.music = temp[80:, :].reshape((140, 272))
        self.validation.labels = temp_l[80:, :].reshape((140, 10))

        self.train.music = temp[:80, :].reshape((560, 272))
        self.train.labels = temp_l[:80, :].reshape((560, 10))
        return self.validation

    def visualize(self, mode='pca'):
        self.one_hot_encode_genres()
        train = self.load_train_data()
        if mode=='pca':
            reduced = reduce.pca(train.music)
        else:
            reduced = reduce.lda(train.music, train.labels)
        x = reduced[:, 0]
        y = reduced[:, 1]
        colors = []
        genres_labels = [None]*len(self.genres)
        for genre in self.genres:
            idx = np.where(self.encoded_genres[genre]==1)[0][0]
            genres_labels[idx]= genre
        for label in train.labels:
            colors.append(np.where(label==1)[0][0])

        std_x = np.std(x)
        std_y = np.std(y)
        plt.scatter(np.divide(x, std_x), np.divide(y, std_y), c=colors, cmap=plt.cm.get_cmap("jet", len(self.genres)))
        cbr = plt.colorbar(ticks=range(len(self.genres)), label='Genres')
        cbr.set_ticklabels(genres_labels)
        if mode=='pca':
            plt.title('PCA projections')
        else:
            plt.title('LDA projection')
        plt.show()

    def destroy_results(self):
        shutil.rmtree(self.results_dir)