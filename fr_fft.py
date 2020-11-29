
import numpy as np

import sounddevice as sd

from tqdm import tqdm

from scipy.signal import get_window

from common import load_config, find_device_id, crop_signal, generate_frequency_range
from common import plot_fft, write_csv

import matplotlib.pyplot as plt

# for debugging:
def plot_sgn(x, fn='wav.png'):

    fig = plt.figure(figsize=(8, 4))
    plt.plot(x)

    plt.savefig(fn)

def generate_sine(freq, n_samples, sr):
    '''
    generate a sine wave for a given frequency, number of samples, and sampling rate

    :param freq: frequency (Hz)
    :param n_samples: number of samples
    :param sr: sampling rate (Hz)
    :return: numpy vector of the generated signal
    '''

    t = np.arange(n_samples)
    return np.sin(2 * np.pi * freq * t / sr)


def get_fft(sgn, sr):

    sgn_len = len(sgn)
    window = get_window('hamming', sgn_len)
    sgn_fft = np.fft.fft(sgn * window) / sgn_len

    timestep = 1/sr
    freqs = np.fft.fftfreq(sgn_fft.size, d=timestep)

    n = len(sgn) // 2

    return freqs[:n], sgn_fft[:n]


def test_fft(freq, sr):
    n_burnin = sr // 2
    sgn_len = sr // 2

    fft_len = int(2**np.floor(np.log2(sgn_len)))
    #fft_len = 2**16

    # generate test wave for the given frequency
    sgn = generate_sine(freq, n_burnin + sgn_len, sr)

    # play & record
    rec = sd.playrec(sgn, samplerate=sr, channels=1)
    sd.wait()

    rec = rec.reshape((len(rec,)))

    plot_sgn(rec, 'rec.png')

    # normalize rec
    rec = rec / np.max(np.abs(rec))

    # crop
    rec = rec[n_burnin:n_burnin+fft_len]
    plot_sgn(rec, 'rec_crop.png')

    sgn = sgn[n_burnin:n_burnin+fft_len]
    plot_sgn(sgn, 'sgn_crop.png')

    # take fft
    fr, am = get_fft(sgn, sr)
    am = np.abs(am).reshape((len(am),))
    plot_fft(fr, am, 'sgn_fft.png')

    fr, am = get_fft(rec, sr)
    am = np.abs(am).reshape((len(am),))
    plot_fft(fr, am, 'rec_fft.png')

    return fr, am



def main():
    cfg = load_config()

    # select device to use
    if 'device' in cfg:
        dev_id = find_device_id(cfg['device'])
        if dev_id is not None:
            sd.default.device = dev_id

    if 'sr' in cfg:
        sr = cfg['sr']
        sd.default.samplerate = sr
    else:
        sr = sd.default.samplerate

    fr, am = test_fft(1000, sr)

    write_csv('fft.csv', fr, am)


if __name__ == '__main__':
    main()
