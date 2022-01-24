import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa
from pathlib import Path
import string
import random

sampling_rate = 44100
downsamp_rate = 16000

input_folder = 'wavfiles'
output_folder = 'clean'


def get_classes():
    classes = {}

    for current_folder, _, files in sorted(os.walk(output_folder)):
        for current_file in files:
            full_path = os.path.join(current_folder, current_file)
            rate, signal = wavfile.read(full_path)

            length = signal.shape[0] / rate

            c = current_folder.rsplit('/', 1)[1]
            if c not in classes.keys():
                classes[c] = length
            else:
                classes[c] = classes[c] + length

    return classes


def plot_classes():
    classes = get_classes()
    class_dist = np.array(list(classes.values()))
    class_labels = np.array(list(classes.keys()))
    class_labels = [label.replace("_", " ") for label in class_labels]

    print(class_dist/np.sum(class_dist)*100)

    fig, ax = plt.subplots()

    cs = cm.Set1(np.arange(40)/40.)

    ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0, 10, 115)))
    ax.pie(class_dist, autopct="%1.1f%%", shadow=False, startangle=90)
    plt.legend(class_labels, loc="upper right")
    plt.show()


def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10),
                       min_periods=1, center=True).mean()

    mask = []
    for mean in y_mean:
        mask.append(True if mean > threshold else False)
    return mask

def main():
    for currentpath, folders, files in tqdm(os.walk(input_folder)):
        for file in files:
            f = os.path.join(currentpath, file)

            signal, rate = librosa.load(f, sr=downsamp_rate)
            mask = envelope(signal, rate, 0.005)

            folder = output_folder + "/" + currentpath[9:]
            if not os.path.exists(folder):
                os.makedirs(folder)

            file = ''.join(random.choices(
                string.ascii_lowercase + string.digits, k=8))

            f = output_folder + "/" + \
                os.path.join(currentpath, file + ".wav")[9:]

            if not os.path.exists(f):
                wavfile.write(filename=f, rate=rate, data=signal[mask])


if __name__ == '__main__':
    main()
