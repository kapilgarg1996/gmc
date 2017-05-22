"""
Module to retrieve features from a single music file.
"""
import librosa
import numpy as np
import pywt

def mfcc(filename, **kwargs):
	"""
	Compute mfcc features for a given music file
	features calculated are : mfcc, delta of mfcc, 
	double delta of mfcc and frame energies
	Instead of frame-wise calculations, use mean, 
	std, max and min of all features.

	Returns complete feature vector
	"""
	y, sr = librosa.load(filename)
	kwargs.setdefault('n_fft', int(sr*0.025))
	kwargs.setdefault('hop_length', int(sr*0.005))
	kwargs.setdefault('n_mels', 128)
	kwargs.setdefault('n_mfcc', 13)
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
	Compute wavelet coefficients
	"""
	y, sr = librosa.load(filename)
	kwargs.setdefault('level', 6)
	dwt_coeffs = pywt.wavedec(y, 'db4', level=int(kwargs['level']))
	mean_features = tuple(np.mean(coeffs, axis=-1) for coeffs in dwt_coeffs)
	std_features = 	tuple(np.std(coeffs, axis=-1) for coeffs in dwt_coeffs)
	max_features = tuple(np.amax(coeffs, axis=-1) for coeffs in dwt_coeffs)
	min_features = tuple(np.amin(coeffs, axis=-1) for coeffs in dwt_coeffs)

	all_features = mean_features + std_features + max_features + min_features
	final_features = np.asarray(all_features)
	return final_features

def beat(filename, **kwargs):
	y, sr = librosa.load(filename)
	kwargs.setdefault('hop_length', 512)
	t, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=int(kwargs['hop_length']))

	return np.hstack((t, beats,))