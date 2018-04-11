import unittest
import os
import numpy as np
import matplotlib.pyplot as plt
from gmc.core.models import nn
from gmc.dataset import musicset, reduce
from gmc.conf import settings
from gmc.core.cache import store

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestNN(unittest.TestCase):
    def test_training(self):
        dataset = musicset.MusicSet(dirname='features_md_6')
        dataset.load_files()
        dataset.load_train_data()
        dataset.load_test_data()
        #print(dataset.test.music.shape)
        #return
        # dataset.load_validation_data()
        # print(dataset.validation.music[:10, :10])
        # return
        if settings.PCA:
            components = reduce.pca_comp(dataset.train.music)
            k = components.shape[1]
            dataset.train.music = np.matmul(components.T, dataset.train.music.T)
            dataset.train.music = dataset.train.music.T
            dataset.test.music = np.matmul(components.T, dataset.test.music.T)
            dataset.test.music = dataset.test.music.T
            nn_t = nn.NN(dataset, n_input=k)
        else:
            nn_t = nn.NN(dataset)
        step = 100
        tr, te, trc, tec = nn_t.train(display_step=step, out=True, path='test.final')
        epochs = np.arange(int(settings.NN['TRAINING_CYCLES']/step))
        f, ax = plt.subplots(2)
        ax[0].plot(epochs, tr, color='#FF2D00', label='Training Accuracy')
        #ax[0].plot(epochs, vl, color='#1DFF00', label='Validation Accuracy')
        ax[0].plot(epochs, te, color='#0014FF', label='Testing Accuracy')
        ax[0].legend(loc="lower right")
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_title('Accuracy vs Number of epochs')
        ax[1].plot(epochs, trc, color='#FF2D00', label='Training Lost')
        ax[1].plot(epochs, tec, color='#1DFF00', label='Test Lost')
        ax[1].legend(loc="lower right")
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss vs Number of Epochs')
        f.suptitle('Network with %s layers' % settings.NN['NUM_HIDDEN_LAYERS'])
        plt.show()

@unittest.skipIf(os.environ.get("DUMMY") == "TRUE",
    "not necessary when real dataset not supplied")
class TestNNSpec(unittest.TestCase):
    def test_nn(self):
        p = os.path.join(settings.BRAIN_DIR, "specfeat")
        dataset = musicset.MusicSet(dirname='special')
        dataset.load_files()
        dataset.one_hot_encode_genres()
        storage = store(p)
        ds = storage['features.dat']
        all_data = np.hstack(tuple(np.split(ds.music, 10, axis=0)))
        all_labels = np.hstack(tuple(np.split(ds.labels, 10, axis=0)))

        dataset.train = musicset.DataSet()
        dataset.test = musicset.DataSet()

        dataset.train.music = np.vstack(tuple(np.split(all_data[:70, :], 10, axis=-1)))
        dataset.train.labels = np.vstack(tuple(np.split(all_labels[:70, :], 10, axis=-1)))

        dataset.test.music = np.vstack(tuple(np.split(all_data[70:, :], 10, axis=-1)))
        dataset.test.labels = np.vstack(tuple(np.split(all_labels[70:, :], 10, axis=-1)))

        nn_t = nn.NN(dataset, n_input=256)
        step = 100
        tr, te, trc, tec = nn_t.train(display_step=step, out=True, path='special.final')
        epochs = np.arange(int(settings.NN['TRAINING_CYCLES']/step))
        f, ax = plt.subplots(2)
        ax[0].plot(epochs, tr, color='#FF2D00', label='Training Accuracy')
        #ax[0].plot(epochs, vl, color='#1DFF00', label='Validation Accuracy')
        ax[0].plot(epochs, te, color='#0014FF', label='Testing Accuracy')
        ax[0].legend(loc="lower right")
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].set_title('Accuracy vs Number of epochs')
        ax[1].plot(epochs, trc, color='#FF2D00', label='Training Lost')
        ax[1].plot(epochs, tec, color='#1DFF00', label='Test Lost')
        ax[1].legend(loc="lower right")
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Loss vs Number of Epochs')
        f.suptitle('Network with %s layers' % settings.NN['NUM_HIDDEN_LAYERS'])
        plt.show()