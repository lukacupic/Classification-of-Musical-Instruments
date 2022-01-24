import os
import scipy
import librosa
import librosa.display
import numpy as np
import pandas as pd
import tensorflow as tf
from statistics import mean
from playsound import playsound
from configparser import ConfigParser
from python_speech_features import mfcc
from python_speech_features import sigproc
import matplotlib.pyplot as plt

config_path = 'config/config.ini'
config = ConfigParser()

classes = ['Acoustic_guitar', 'Bass_drum', 'Cello', 'Clarinet', 'Double_bass', 'Flute',
           'Harmonica', 'Hi-hat', 'Saxophone', 'Snare_drum', 'Ukulele', 'Violin_or_fiddle']

y_label = 'mel'


def get_prop(name):
    value = config.get('main', name)

    try:
        return int(value)
    except ValueError:
        return float(value)


def set_prop(name, value):
    config.set('main', name, str(value))


def plot_spectrogram_linear(S, sr, hop_length):
    plt.figure(figsize=(10, 4))
    D = np.abs(S)
    librosa.display.specshow(D, x_axis='time', y_axis=y_label,
                             cmap='Blues', sr=sr, hop_length=hop_length)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_spectrogram_log(S, sr, hop_length):
    plt.figure(figsize=(10, 4))
    D = np.abs(S)
    D_dB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(
        D_dB, y_axis=y_label, x_axis='time', cmap='Blues', sr=sr, hop_length=hop_length)
    #plt.title('Decibel spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_melspectrogram(S, sr, hop_length):
    plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time', y_axis=y_label, cmap='Blues',
                             sr=sr, hop_length=hop_length)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.tight_layout()
    plt.show()


def plot_mfccs(mfccs, sr, hop_length):
    librosa.display.specshow(mfccs, x_axis='time',
                             cmap='Blues', sr=sr, hop_length=hop_length)
    # plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_filterbank(rate, nfft, n_mels):
    mels = librosa.filters.mel(
        sr=16000, n_fft=8000, n_mels=15)
    mels /= np.max(mels, axis=-1)[:, None]
    plt.plot(mels.T)
    plt.xlabel('Frequency [Hz]')
    plt.show()


def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10),
                       min_periods=1, center=True).mean()

    mask = []
    for mean in y_mean:
        mask.append(True if mean > threshold else False)
    return mask


def load_file(f):
    signal, rate = librosa.load(f, sr=get_prop('rate'))
    mask = envelope(signal, rate, 0.0005)
    return signal[mask], rate


def main():
    step = get_prop('step')
    nmfcc = get_prop('nmfcc')
    hop_length = get_prop('hop_length')
    n_fft = get_prop('nfft')
    _min = get_prop('min')
    _max = get_prop('max')

    wav, rate = load_file('examples/vietnam.wav')
    # sample = wav[1600: 3200]
    sample = wav

    # plot_filterbank(0, 0, 0)

    S = librosa.stft(y=sample, n_fft=n_fft, hop_length=hop_length)
    
    # plot_spectrogram_linear(S, sr=rate, hop_length=hop_length)

    plot_spectrogram_log(S, sr=rate, hop_length=hop_length)

    # S = librosa.feature.melspectrogram(
    #     y=sample, sr=rate, S=S, n_fft=n_fft, hop_length=hop_length)
    # plot_melspectrogram(S=S, sr=rate, hop_length=hop_length)

    # mfccs = scipy.fftpack.dct(librosa.power_to_db(
    #     S), axis=0, type=2, norm='ortho')[:nmfcc]
    # plot_mfccs(mfccs=mfccs, sr=rate, hop_length=hop_length)


if __name__ == "__main__":
    config.read(config_path)
    main()
