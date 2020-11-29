
import numpy as np
import sounddevice as sd

import matplotlib.pyplot as plt
import csv

from tqdm import tqdm

from common import load_config, find_device_id

# for debugging:
def plot_sgn(x, fn='wav.png'):

    fig = plt.figure(figsize=(8,4))
    #librosa.display.waveplot(x, sr=sr)
    plt.plot(x)

    plt.savefig(fn)


def generate_sine(freq, sgn_len, sr):
    '''
    generate a sine wave for a given frequency, length, and sampling rate

    :param freq: frequency (Hz)
    :param sgn_len: length (seconds)
    :param sr: sampling rate (Hz)
    :return: numpy vector
    '''

    ns = int(sgn_len * sr)
    t = np.arange(ns)
    return np.sin(2 * np.pi * freq * t / sr)


def crop_signal(x, drop_begin, drop_end, sr):
    '''
    drop out pieces from the beginning and the end of a signal

    :param x: signal to crop
    :param drop_begin: beginning dropout length (seconds)
    :param drop_end: end dropout length (seconds)
    :param sr: sample rate
    :return: cropped signal
    '''

    start_ind = int(drop_begin * sr)
    end_ind = len(x) - int(drop_end * sr)

    return x[start_ind:end_ind]


def test_frequency(freq, sr, config):
    '''
    test a single frequency

    :param freq: frequency to test
    :param sr: sample rate
    :param config: config dict
    :return: amplitude, rms
    '''

    n_waves = config.get('n_waves', 20)
    drop_begin = config.get('drop_begin', 0.0)
    drop_end = config.get('drop_end', 0.0)

    # generate test signal
    sgn_len = n_waves / freq + drop_begin + drop_end
    test_signal = generate_sine(freq, sgn_len, sr)

    # play/rec
    rec_signal = sd.playrec(test_signal, channels=1)

    # block execution
    sd.wait()

    # do the cropping
    rec_signal = crop_signal(rec_signal, drop_begin, drop_end, sr)

    # return values
    ampl = np.max(np.abs(rec_signal))
    rms = np.mean(np.square(rec_signal))

    return ampl, rms



def generate_frequency_range(f0, f1, points_in_decade):
    '''
    generate logarithmic range of points. endpoints are included

    :param f0: start frequency
    :param f1: end frequency
    :param points_in_decade: number of points in a decade
    :return: numpy vector
    '''

    if f0 > f1:
        return None

    n_decades = np.log10(f1 / f0)
    n_points = n_decades * points_in_decade
    b = np.log10(f0)
    k = n_decades * np.arange(n_points + 1) / n_points + b

    return 10**k


def plot_frequency_response(fr, amp, fn='fr.png'):
    fig = plt.figure(figsize=(14,4))
    plt.semilogx(fr, 20 * np.log10(amp))
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.grid(True)
    plt.savefig(fn)

def write_csv(csv_fn, freqs, ampls):
    with open(csv_fn, 'wt') as f:
        csv_writer = csv.writer(f)
        hdr = ['f', 'a']

        csv_writer.writerow(hdr)

        for row in zip(freqs, ampls):
            csv_writer.writerow(row)


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

    # get reference amplitude for normalization
    ref_ampl, _ = test_frequency(1000, sr, cfg)

    print('normalizing factor {}'.format(ref_ampl))

    f0 = cfg.get('f0', 10)
    f1 = cfg.get('f1', 20000)
    pid = cfg.get('points_in_decade', 5)

    freqs = generate_frequency_range(f0, f1, pid)
    ampls = []

    for f in tqdm(freqs):
        amplitude, rms = test_frequency(f, sr, cfg)
        ampls.append(amplitude)

    ampls = np.array(ampls) / ref_ampl

    plot_frequency_response(freqs, ampls)

    if 'csv_fn' in cfg:
        write_csv(cfg['csv_fn'], freqs, ampls)

if __name__ == '__main__':
    main()