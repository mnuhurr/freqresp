
import numpy as np

import sounddevice as sd

from tqdm import tqdm

from scipy.signal import get_window

from common import load_config, find_device_id, crop_signal, generate_frequency_range
from common import plot_frequency_response, write_csv

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
    '''
    generate frequency bins for given fft length and sample rate

    :param fft_len: fft length
    :param sr: sample rate
    :return: vector of length fft_len containing the corresponding frequencies
    '''

    timestep = 1/sr
    freqs = np.fft.fftfreq(fft_len, d=timestep)

    n = fft_len // 2

    return freqs[:n]

def get_fft(sgn, sr):
    '''
    compute fft of a given signal. currently uses hamming window.

    :param sgn: signal
    :param sr:  sample rate
    :return:
    '''
    # windowing
    sgn_len = len(sgn)
    window = get_window('hamming', sgn_len)
    sgn_win = sgn * window

    # normalize
    #sgn_win /= np.max(np.abs(sgn_win))

    # fft
    sgn_fft = np.fft.fft(sgn_win) / sgn_len

    n = sgn_len // 2

    return sgn_fft[:n]


def test_fft(freq, sr, fft_len=4096):
    '''
    generate a test signal of single sine wave, record the output and compute fft of the recorded signal. returns two
    vectors: first one contains frequencies, the second amplitudes

    :param freq: test frequency
    :param sr: sample rate
    :param fft_len: fft length
    :return: frequency, amplitude
    '''

    n_burnin = sr // 2
    sgn_len = sr // 2

    # generate test wave for the given frequency
    sgn = generate_sine(freq, n_burnin + sgn_len, sr)

    # play & record
    rec = sd.playrec(sgn, samplerate=sr, channels=1)[:, 0]
    sd.wait()

    # crop
    rec = rec[n_burnin:n_burnin + fft_len]

    # do the fft
    fr = fft_freqs(len(rec), sr)
    am = get_fft(rec, sr)
    am = np.abs(am)

    return fr, am


def thd(am, f_ind):
    '''
    compute THD from the FFT results. assumes that the system input signal frequency is exactly one of the FFT bin
    frequencies.

    :param am: amplitudes
    :param f_ind: input signal index
    :return: THD percentage
    '''

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

    # read some settings
    fft_cfg = cfg.get('fft', {})

    fft_len = fft_cfg.get('fft_len', 2**12)
    f = fft_cfg.get('freq', 1000)

    # get fft frequencies for the used parameters
    freqs = fft_freqs(fft_len, sr)

    # select the nearest one
    f_ind = np.argmin(np.abs(freqs - f))

    # run the test
    fr, am = test_fft(freqs[f_ind], sr, fft_len=fft_len)


    # save results
    if 'plot_filename' in fft_cfg:
        plot_frequency_response(fr, am, fft_cfg['plot_filename'])

    if 'csv_filename' in fft_cfg:
        write_csv(fft_cfg['csv_filename'], fr, am)


    # compute thd
    thd_pct = thd(am, f_ind)
    thd_db = 20 * np.log10(thd_pct)
    print('thd: {:.2f} dB'.format(thd_db))


if __name__ == '__main__':
    main()
