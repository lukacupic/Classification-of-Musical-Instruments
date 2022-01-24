import os
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


def get_prop(name):
    value = config.get('main', name)

    try:
        return int(value)
    except ValueError:
        return float(value)


def set_prop(name, value):
    config.set('main', name, str(value))


def predict(rate, wav, classes, interpreter_info):
    y_probs = []

    step = get_prop('step')
    nmfcc = get_prop('nmfcc')
    hop_length = get_prop('hop_length')
    nfft = get_prop('nfft')
    _min = get_prop('min')
    _max = get_prop('max')

    for i in range(0, wav.shape[0] - step, step):
        sample = wav[i: i + step]

        x = librosa.feature.mfcc(
            y=sample, sr=rate, n_fft=nfft, hop_length=hop_length, n_mfcc=nmfcc)
            
        x = (x - _min) / (_max - _min)
        x = x.reshape(1, x.shape[0], x.shape[1], 1)

        y_ = run_interpreter(
            x, interpreter_info[0], interpreter_info[1], interpreter_info[2], interpreter_info[3])
        y_probs.append(y_)

    return np.mean(y_probs, axis=0).flatten()


def envelope(y, rate, threshold):
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10),
                       min_periods=1, center=True).mean()

    mask = []
    for mean in y_mean:
        mask.append(True if mean > threshold else False)
    return mask


def get_classes(folder):
    folders = [f[0] for f in sorted(os.walk(folder)) if f[0] is not folder]
    classes = [f[len(folder)+1:] for f in folders]
    return classes


def load_file(f):
    signal, rate = librosa.load(f, sr=get_prop('rate'))
    mask = envelope(signal, rate, 0.0005)
    return signal[mask], rate


def load_mic_data(file):
    wav = []
    with open(file) as f:
        for line in f:
            for value in line.split():
                wav.append(float(value))

    wav = np.asfortranarray(wav)

    return wav, get_prop('rate')


def load_interpreter(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    return [interpreter, input_details, output_details, input_shape]


def run_interpreter(x, interpreter, input_details, output_details, input_shape):
    input_data = np.array(x, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


def main():
    classes = get_classes('clean')
    wav, rate = load_file("examples/uke.wav")

    interpreter_info = load_interpreter("models/model.tflite")

    prob = predict(rate, wav, classes, interpreter_info)
    print(classes[np.argmax(prob)], np.max(prob))


if __name__ == "__main__":
    config.read(config_path)

    main()
