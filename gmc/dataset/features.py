"""
Module to retrieve features from a single music file.
"""
import librosa
import numpy as np
import pywt
import tensorflow as tf
from gmc.conf import settings

def mfcc(filename, **kwargs):
    """
    Compute mfcc features for a given music file
    features calculated are : mfcc, delta of mfcc, 
    double delta of mfcc and frame energies
    Instead of frame-wise calculations, use mean, 
    std, max and min of all features.

    Returns complete feature vector (160 features)
    """
    y, sr = librosa.load(filename)
    kwargs.setdefault('n_fft', int(sr*settings.FRAME_LENGTH))
    kwargs.setdefault('hop_length', int(sr*settings.HOP_LENGTH))
    kwargs.setdefault('n_mels', 128)
    kwargs.setdefault('n_mfcc', settings.N_MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, **kwargs)
    deltas = librosa.feature.delta(mfcc)
    d_deltas = librosa.feature.delta(mfcc, order=2)
    
    y_ = np.pad(y, int(kwargs['n_fft']//2), mode='reflect')
    energies = librosa.feature.rmse(y=y_, frame_length=int(kwargs['n_fft']), 
        hop_length=int(kwargs['hop_length']))

    all_features = np.vstack((mfcc, deltas, d_deltas, energies,))
    
    mean_features = np.mean(all_features, axis=-1)
    std_features = np.std(all_features, axis=-1)
    max_features = np.amax(all_features, axis=-1)
    min_features = np.amin(all_features, axis=-1)

    final_features = np.vstack((mean_features, std_features, 
        max_features, min_features,))

    return np.ravel(final_features)

def dwt(filename, **kwargs):
    """
    Compute wavelet coefficients frame wise and return computed features
    (112 features)
    """
    y, sr = librosa.load(filename)
    options = {}
    options['frame_length'] = int(sr*settings.FRAME_LENGTH*settings.W_FRAME_SCALE)
    options['hop_length'] = int(sr*settings.HOP_LENGTH*settings.W_FRAME_SCALE)
    frames = librosa.util.frame(y, **options)
    frames = frames.T
    wavelets = None
    kwargs.setdefault('level', 6)
    
    for frame in frames:
        dwt_coeffs = pywt.wavedec(frame, 'db4', level=int(kwargs['level']))
        mean_features = tuple(np.mean(coeffs, axis=-1) for coeffs in dwt_coeffs)
        std_features =  tuple(np.std(coeffs, axis=-1) for coeffs in dwt_coeffs)
        max_features = tuple(np.amax(coeffs, axis=-1) for coeffs in dwt_coeffs)
        min_features = tuple(np.amin(coeffs, axis=-1) for coeffs in dwt_coeffs)

        all_features = mean_features + std_features + max_features + min_features
        final_features = np.asarray(all_features)

        if wavelets is None:
            wavelets = np.array(final_features)
        else:
            wavelets = np.vstack((wavelets, final_features,))

    all_features = wavelets.T
    mean_features = np.mean(all_features, axis=-1)
    std_features = np.std(all_features, axis=-1)
    max_features = np.amax(all_features, axis=-1)
    min_features = np.amin(all_features, axis=-1)

    final_features = np.vstack((mean_features, std_features, 
        max_features, min_features,))

    return np.ravel(final_features)

def beat(filename, **kwargs):
    """
    Compute beats locations and find a relation among those locations.
    Use a simple linear regressor to find W and b for beat locations
    Return a feature vector (11 features)
    """
    y, sr = librosa.load(filename)
    kwargs.setdefault('hop_length', 512)
    t, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=int(kwargs['hop_length']))

    # rng = np.random
    # train_X = np.arange(1,len(beats)+1)
    # train_Y = beats
    # n_samples = train_X.shape[0]

    # # tf Graph Input
    # X = tf.placeholder("float")
    # Y = tf.placeholder("float")

    # # Set model weights
    # W = tf.Variable(rng.randn(), name="weight")
    # b = tf.Variable(rng.randn(), name="bias")

    # # Construct a linear model
    # pred = tf.add(tf.multiply(X, W), b)

    # # Mean squared error
    # cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
    # # Gradient descent
    # #  Note, minimize() knows to modify W and b because Variable objects are trainable=True by default
    # optimizer = tf.train.GradientDescentOptimizer(0.02).minimize(cost)

    # # Initializing the variables
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     for epoch in range(50):
    #         for (x, y) in zip(train_X, train_Y):
    #             sess.run(optimizer, feed_dict={X: x, Y: y})

    # features = np.array([t, W, b])
    first_frames = beats[:10]

    return np.hstack((t, first_frames,))