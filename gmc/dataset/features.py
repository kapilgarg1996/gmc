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
    dur = kwargs.get('duration', None)
    y, sr = librosa.load(filename, duration=dur)
    kwargs.setdefault('n_fft', int(sr*settings.FRAME_LENGTH))
    kwargs.setdefault('hop_length', int(sr*settings.HOP_LENGTH))
    kwargs.setdefault('n_mels', 128)
    kwargs.setdefault('n_mfcc', settings.N_MFCC)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, **kwargs)
    all_features = np.array(mfcc)
    if 'delta' in settings.MFCC_EXTRA:
        deltas = librosa.feature.delta(mfcc)
        all_features = np.vstack((all_features, deltas,))
    if 'ddelta' in settings.MFCC_EXTRA:
        d_deltas = librosa.feature.delta(mfcc, order=2)
        all_features = np.vstack((all_features, d_deltas,))
    if 'energy' in settings.MFCC_EXTRA:
        y_ = np.pad(y, int(kwargs['n_fft']//2), mode='reflect')
        energies = librosa.feature.rmse(y=y_, frame_length=int(kwargs['n_fft']), 
            hop_length=int(kwargs['hop_length']))
        all_features = np.vstack((all_features, energies,))

    final_features = None

    if settings.KEEP_FRAMES > 0:
        num_batch = int(all_features.shape[1]/settings.KEEP_FRAMES)
        for i in range(num_batch):
            batch = all_features[:, i*settings.KEEP_FRAMES:(i+1)*settings.KEEP_FRAMES]
            mean_features = np.mean(batch, axis=-1)
            std_features = np.std(batch, axis=-1)
            max_features = np.amax(batch, axis=-1)
            min_features = np.amin(batch, axis=-1)
            total_features = np.vstack((mean_features, std_features, 
                max_features, min_features,))
            if final_features is None:
                final_features = np.ravel(total_features)
            else:
                final_features = np.vstack((final_features, np.ravel(total_features)))
    else:
        batch = all_features
        mean_features = np.mean(batch, axis=-1)
        std_features = np.std(batch, axis=-1)
        max_features = np.amax(batch, axis=-1)
        min_features = np.amin(batch, axis=-1)
        total_features = np.vstack((mean_features, std_features, 
            max_features, min_features,))
        final_features = np.ravel(total_features)
    return final_features

def dwt(filename, **kwargs):
    """
    Compute wavelet coefficients frame wise and return computed features
    (112 features)
    """
    dur = kwargs.get('duration', None)
    y, sr = librosa.load(filename, duration=dur)
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
    final_features = None

    if settings.KEEP_FRAMES > 0:
        num_batch = int(all_features.shape[1]/settings.KEEP_FRAMES)
        for i in range(num_batch):
            batch = all_features[:, i*settings.KEEP_FRAMES:(i+1)*settings.KEEP_FRAMES]
            mean_features = np.mean(batch, axis=-1)
            std_features = np.std(batch, axis=-1)
            max_features = np.amax(batch, axis=-1)
            min_features = np.amin(batch, axis=-1)
            total_features = np.vstack((mean_features, std_features, 
                max_features, min_features,))
            if final_features is None:
                final_features = np.ravel(total_features)
            else:
                final_features = np.vstack((final_features, np.ravel(total_features)))
    else:
        batch = all_features
        mean_features = np.mean(batch, axis=-1)
        std_features = np.std(batch, axis=-1)
        max_features = np.amax(batch, axis=-1)
        min_features = np.amin(batch, axis=-1)
        total_features = np.vstack((mean_features, std_features, 
            max_features, min_features,))
        final_features = np.ravel(total_features)
    return final_features

def beat(filename, **kwargs):
    """
    Compute beats locations and find a relation among those locations.
    Use a simple linear regressor to find W and b for beat locations
    Return a feature vector (11 features)
    """
    dur = kwargs.get('duration', None)
    y, sr = librosa.load(filename, duration=dur)
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
    first_frames = beats[:settings.NUM_BEATS]

    return np.hstack((t, first_frames,))