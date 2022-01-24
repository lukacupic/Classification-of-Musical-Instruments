import os
import pickle
import librosa
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
import tensorflow as tf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.metrics import f1_score
from configparser import ConfigParser
from keras.utils import to_categorical
from python_speech_features import mfcc
from sklearn.metrics import recall_score
from python_speech_features import sigproc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.layers import Dropout, Dense, TimeDistributed
from sklearn.utils.class_weight import compute_class_weight

config = ConfigParser()
config_path = 'config/config.ini'
config_path_android = '../Android/app/src/main/assets/config.ini'
model_path_android = '../Android/app/src/main/assets/model.tflite'


def get_prop(name):
    value = config.get('main', name)

    try:
        return int(value)
    except ValueError:
        return float(value)


def set_prop(name, value):
    config.set('main', name, str(value))


def get_conv_model(input_shape):
    model = Sequential()

    kernel_size = (3, 3)
    stride = (1, 1)
    activation = 'relu'
    padding = 'same'

    model.add(Conv2D(16, kernel_size, activation=activation,
                     strides=stride, padding=padding, input_shape=input_shape))
    model.add(Conv2D(32, kernel_size, activation=activation,
                     strides=stride, padding=padding))
    model.add(Conv2D(64, kernel_size, activation=activation,
                     strides=stride, padding=padding))
    model.add(Conv2D(128, kernel_size, activation=activation,
                     strides=stride, padding=padding))

    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(128, activation=activation))
    model.add(Dense(64, activation=activation))
    model.add(Dense(get_prop('no_classes'), activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def get_model(X):
    input_shape = (X.shape[1], X.shape[2], 1)
    return get_conv_model(input_shape)


def save_model(model):
    model.save('models/model.h5')

    model = tf.keras.models.load_model('models/model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite = converter.convert()
    open('models/model.tflite', 'wb').write(tflite)
    open(model_path_android, 'wb').write(tflite)


def build_features(n_samples, prob_dist, classes):
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    step = get_prop('step')
    nmfcc = get_prop('nmfcc')
    hop_length = get_prop('hop_length')
    nfft = get_prop('nfft')

    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(classes, p=prob_dist)
        file = np.random.choice(os.listdir('clean/' + rand_class))

        rate, signal = wavfile.read('clean/' + rand_class + '/' + file)
        signal = signal / 2.5

        rand_index = np.random.randint(0, signal.shape[0] - step)
        sample = signal[rand_index:rand_index + step]

        x = librosa.feature.mfcc(
            y=sample, sr=rate, n_fft=nfft, hop_length=hop_length, n_mfcc=nmfcc)

        _min = min(np.amin(x), _min)
        _max = max(np.amax(x), _max)

        X.append(x)
        y.append(classes.index(rand_class))

    set_prop('min', _min)
    set_prop('max', _max)

    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)

    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

    y = to_categorical(y, num_classes=len(classes))

    return X, y


def get_features():
    classes = {}

    if os.path.exists('config/features.p'):
        return pickle.load(open('config/features.p', 'rb'))

    for current_folder, _, files in sorted(os.walk('clean')):
        for current_file in files:
            full_path = os.path.join(current_folder, current_file)
            rate, signal = wavfile.read(full_path)

            length = signal.shape[0] / rate

            c = current_folder.rsplit('/', 1)[1]
            if c not in classes.keys():
                classes[c] = length
            else:
                classes[c] = classes[c] + length

    class_dist = np.array(list(classes.values()))
    prob_dist = class_dist / sum(class_dist)
    n_samples = 5 * int(sum(class_dist) / get_prop('chunk_length'))

    set_prop('no_classes', len(class_dist))

    class_labels = list(classes.keys())

    X, y = build_features(n_samples, prob_dist, class_labels)
    pickle.dump((X, y, class_labels), open('config/features.p', 'wb'))

    class_labels = [label.replace("_", " ") for label in class_labels]
    return X, y, class_labels


def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def plot_conf_matrix(y, y_pred, classes):
    cm = confusion_matrix(y_true=y, y_pred=y_pred)

    df_cm = pd.DataFrame(cm, classes, classes)
    sn.set(font_scale=1.0)
    sn.heatmap(df_cm, cmap='Blues', annot=True, annot_kws={
               'size': get_prop('no_classes')}, fmt='d', cbar=False)
    plt.show()


def save_config():
    # write to Python directory
    with open(config_path, 'w') as f:
        config.write(f)

    # write to Android directory
    with open(config_path_android, 'w') as f:
        config.write(f)


def get_metrics(y_true, y_pred, classes):
    plot_conf_matrix(y_true, y_pred, classes)

    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=None)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average=None)
    recall = recall_score(y_true=y_true, y_pred=y_pred, average=None)

    for i in range(0, 12):
        digits = 6
        a = str(precision[i])[0:digits]
        b = str(recall[i])[0:digits]
        c = str(f1[i])[0:digits]

        print(12 * ' ' + classes[i] + ' & ' + a +
              ' & ' + b + ' & ' + c + ' & \\\\')
        print(8 * ' ' + '\hline')


def main():
    config.read(config_path)

    X, y, classes = get_features()
    y_ = np.argmax(y, axis=1)
    class_weight = compute_class_weight('balanced', np.unique(y_), y_)

    if os.path.exists('models/model.h5'):
        model = tf.keras.models.load_model('models/model.h5')
    else:
        model = get_model(X)
        history = model.fit(X, y, validation_split=0.33, epochs=10,
                            batch_size=32, shuffle=True, class_weight=class_weight)
        plot_history(history)
        save_model(model)

    y_pred = model.predict(X)
    classes = [label.replace("_", " ") for label in classes]
    get_metrics(y_true=y.argmax(axis=1), y_pred=y_pred.argmax(axis=1), classes=classes)
    model.summary()

    save_config()


if __name__ == '__main__':
    main()