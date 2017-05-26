DATASET_DIR = '/tmp'

BRAIN_DIR = '/tmp'

GENRES = [
    'blues', 'classical', 'country', 'disco', 'hiphop',
    'jazz', 'metal', 'pop', 'reggae', 'rock'
]

TRAIN_TEST_RATIO = [7, 3]

MODE = 'nn'

PCA = False

FEATURES = ['mfcc', 'dwt', 'beat']
FEATURES_LENGTH = {
    'mfcc' : 160,
    'dwt' : 112,
    'beat' : 13
}

FRAME_LENGTH = 0.025
HOP_LENGTH = 0.005
N_MFCC = 13
W_FRAME_SCALE = 10

NN = {
    'NUM_HIDDEN_LAYERS' : 2,
    'HIDDEN_INPUTS' : [1024, 1024],
    'RANDOM' : True,
    'BATCH_SIZE' : 100,
    'TRAINING_CYCLES' : 1000,
    'LEARNING_RATE' : 0.01,
    'DROPOUT_PROB' : 0.6
}

CNN = {
    'NUM_HIDDEN_LAYERS' : 2,
    'HIDDEN_FEATURES' : [32, 64],
    'PATCH_SIZE' : [5, 5],
    'RANDOM' : False,
    'STRIDES' : [1, 1, 1, 1],
    'BATCH_SIZE' : 100,
    'TRAINING_CYCLES' : 1000,
    'LEARNING_RATE' : 0.01,
    'DROPOUT_PROB' : 0.6
}