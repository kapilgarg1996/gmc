import sys
import tkinter as tk

import os
import matplotlib
import tensorflow as tf
import numpy as np
import pyaudio
import wave
import subprocess
from PIL import ImageTk, Image
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from gmc.dataset.utils import mel_spec_plot as msp
from gmc.core.models import nn
from gmc.dataset import features, musicset
from gmc.core.cache import store
from gmc.conf import settings
from gmc.core import handler

class GmcApp:
    def __init__(self, root):
        self.root = root
        self.img = tk.PhotoImage(file="icon.gif")
        self.canvas = tk.Label(root, image=self.img)
        self.canvas.image = self.img
        self.canvas.grid(row=0, column=0)
        self.prediction = None
        root.wm_title("Music Classifier")
        tk.Button(root, text = "Select File", command = lambda: self.openFile()).grid(row=1, column=0, pady=5)
        tk.Button(root, text = "Record Audio", command = lambda: self.record()).grid(row=1, column=1, pady=5)
        tk.Button(root, text = "Classify", command = lambda: self.classify()).grid(row=1, column=2, pady=5)

    def plot (self, filepath):
        if self.prediction is not None:
            self.prediction.destroy()
        if self.canvas is not None:
            self.canvas.destroy()
        fig = msp(filepath)
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        self.canvas = canvas.get_tk_widget()
        self.canvas.grid(row=0, column=0)
        canvas.draw()

    def openFile(self):
        self.fileName = askopenfilename(initialdir = ".")
        self.plot(self.fileName)

    def classify(self):
        storage = store(os.path.join(settings.BRAIN_DIR, 'nn'))
        save_path = storage['save.path']
        meta_path = save_path + '.meta'
        saver = tf.train.import_meta_graph(meta_path)
        data = self.get_features()
        n_f = data.shape[0]
        data = data.reshape((1, n_f))
        prediction = None
        with tf.Session() as sess:
            saver.restore(sess, save_path)
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name('x:0')
            y_ = graph.get_tensor_by_name('y_:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            result = sess.run(y_, feed_dict={x : data, keep_prob:1})[0]
            idx = np.argmax(result)
            dataset = musicset.MusicSet()
            dataset.one_hot_encode_genres()
            for genre in dataset.genres:
                if dataset.encoded_genres[genre][idx] == 1:
                    prediction = genre

        if self.prediction is not None:
            self.prediction.destroy()

        self.prediction = tk.Label(self.root, text=prediction)
        self.prediction.config(font=("Courier", 36))
        self.prediction.grid(row=0, column=1)

    def get_features(self):
        result = None
        for f in settings.FEATURES:
            feat_func = getattr(features, f)
            if result is None:
                result = feat_func(self.fileName)
            else:
                result = np.hstack((result, feat_func(self.fileName)))
        return result

    def record(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 22050
        CHUNK = 1024
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = os.path.join(settings.BRAIN_DIR, "file.wav")
        FINAL_OUTPUT_FILENAME = os.path.join(settings.BRAIN_DIR, "output.wav")
        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print("recording...")
        frames = []
         
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("finished recording")
         
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        p = subprocess.Popen(['ffmpeg', '-y', '-i', WAVE_OUTPUT_FILENAME, 
            '-map_channel', '0.0.0', FINAL_OUTPUT_FILENAME], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_out, p_err = p.communicate()
        self.fileName = FINAL_OUTPUT_FILENAME
        self.plot(FINAL_OUTPUT_FILENAME)

def show_app():
    handler.execute_from_command_line()
    root = tk.Tk()
    app = GmcApp(root)
    root.protocol('WM_DELETE_WINDOW', lambda: close(app, root))  # root is your root window
    root.mainloop()

def close(app, root):
    plt.close()
    root.destroy()

if __name__ == '__main__':
    show_app()