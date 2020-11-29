
import numpy as np

import sounddevice as sd

from tqdm import tqdm

from scipy.signal import get_window

from common import load_config, find_device_id, crop_signal, generate_frequency_range
from common import plot_frequency_response, plot_fft, write_csv

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


def fft_freqs(fft_len, sr):
    timestep = 1/sr
    freqs = np.fft.fftfreq(fft_len, d=timestep)

    return freqs

def get_fft(sgn, sr):

    # windowing
    sgn_len = len(sgn)
    window = get_window('hamming', sgn_len)
    sgn_win = sgn * window

    # normalize
    #sgn_win /= np.max(np.abs(sgn_win))

    # fft
    sgn_fft = np.fft.fft(sgn_win) / sgn_len

    # frequency bins
    freqs = fft_freqs(sgn_len, sr)

    n = sgn_len // 2

    return freqs[:n], sgn_fft[:n]


def test_fft(freq, sr, fft_len=4096):
    n_burnin = sr // 2
    sgn_len = sr // 2

    # generate test wave for the given frequency
    sgn = generate_sine(freq, n_burnin + sgn_len, sr)

    # play & record
    rec = sd.playrec(sgn, samplerate=sr, channels=1)[:, 0]
    sd.wait()

    # crop
    rec = rec[n_burnin:n_burnin + fft_len]
    plot_sgn(sgn, 'sgn_crop.png')

    # do the fft
    fr, am = get_fft(rec, sr)
    am = np.abs(am).reshape((len(am),))
    plot_fft(fr, am, 'rec_fft.png')

    return fr, am


def thd(am, f_ind):
    inds = np.arange(f_ind, len(am), f_ind)
    vn = am[inds]

    # https://www.ti.com/lit/an/slaa114/slaa114.pdf
    # correction factors for windows:
    # blackman-harris: 0.7610
    # hanning: 0.8165
    # hamming: 0.8566
    # bartlett: 0.8660

    return np.sqrt(np.sum(np.square(vn[1:]))) / vn[0]




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

    fft_len = 2**12
    f = 1000

    # get fft frequencies for the used parameters
    freqs = fft_freqs(fft_len, sr)

    # select the nearest one
    f_ind = np.argmin(np.abs(freqs - f))

    fr, am = test_fft(freqs[f_ind], sr)


    #plot_fft(fr, am, 'fft.png')
    plot_frequency_response(fr, am, 'fft.png')
    write_csv('fft.csv', fr, am)

    # thd
    thd_pct = thd(am, f_ind)
    thd_db = 20 * np.log10(thd_pct)
    print('thd: {:.2f} dB'.format(thd_db))


if __name__ == '__main__':
    main()
