import librosa
import numpy as np
from librosa import display
import matplotlib.pyplot as plt

def mel_spec_plot(filepath):
    y, sr = librosa.load(filepath)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', 
            fmax=8000, x_axis='time')
    f = plt.gcf()
    return f
