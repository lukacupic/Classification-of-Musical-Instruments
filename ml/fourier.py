import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def plot_signal(y, sr):
    plt.figure()
    plt.subplot(1, 1, 1)
    librosa.display.waveplot(y, sr=sr)

def plot_fourier_new(signal, dt):
    fourier = np.fft.rfft(signal)
    freq = np.fft.fftfreq(len(signal))

    fig, ax = plt.subplots()
    ax.plot(freq, fourier)
    plt.show()


def plot_fourier(sig, dt):
    t = np.arange(0, sig.shape[-1]) * dt

    sigFFT = np.fft.fft(sig) / t.shape[0]
    freq = np.fft.fftfreq(t.shape[0], d=dt)

    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    sigFFTPos = 2 * sigFFT[0:firstNegInd]

    sigFFTPos = librosa.util.normalize(sigFFTPos)

    plt.figure(figsize=(10, 4))
    plt.plot(freqAxisPos, np.abs(sigFFTPos))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.show()


if __name__ == "__main__":
    # signal, rate = librosa.load("examples/guitar.wav")
    # signal = librosa.util.normalize(signal)
    # #plot_signal(signal, rate)
    # plot_fourier(signal, dt=1/rate)

    signal, rate = librosa.load("examples/vietnam.wav")
    signal = librosa.util.normalize(signal)
    #plot_signal(signal, rate)
    plot_fourier(signal, dt=1/rate)