'''

simple tool to measure frequency response. generates several sinewaves and measures the maximum amplitude of the
response. more sophisticated methods to come.

'''

import numpy as np
import sounddevice as sd

from tqdm import tqdm

from common import load_config, find_device_id, crop_signal, generate_frequency_range
from common import plot_frequency_response, write_csv

from filtering import bandpass, lowpass

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
    rec_signal = sd.playrec(test_signal, samplerate=sr, channels=1)

    # block execution
    sd.wait()

    # do the cropping
    rec_signal = crop_signal(rec_signal, drop_begin, drop_end, sr)

    # filtering. do some normalization due to that
    #rec_signal = bandpass(rec_signal, freq, sr)

    # return values
    ampl = np.max(np.abs(rec_signal))
    rms = np.mean(np.square(rec_signal))

    return ampl, rms


def normalizing_factor(config, sr):
    '''
    get normalizing settings. get a reference measurement if needed.

    :param config:
    :return:
    '''

    ref_ampl = 1
    if 'sweep' in config and 'normalize' in config['sweep']:
        if 'frequency' in config['sweep']['normalize']:
            f_norm = config['sweep']['normalize']['frequency']
            print('testing freq', f_norm)
            ref_ampl, _ = test_frequency(f_norm, sr, config)

        elif 'factor' in config['sweep']['normalize']:
            ref_ampl = config['sweep']['normalize']['factor']

    return ref_ampl


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

    repeats = 10

    # get reference amplitude for normalization
    nf = normalizing_factor(cfg, sr)

    # debug prints
    print('using normalizing factor {}'.format(nf))

    sweep_cfg = cfg.get('sweep', {})
    f0 = sweep_cfg.get('f0', 10)
    f1 = sweep_cfg.get('f1', 10000)
    pid = sweep_cfg.get('points_in_decade', 5)

    freqs = generate_frequency_range(f0, f1, pid)
    ams = []

    for r in range(repeats):
        ampls = []
        print('round {}/{}'.format(r + 1, repeats))

        for f in tqdm(freqs):
            amplitude, rms = test_frequency(f, sr, cfg)
            ampls.append(amplitude)

        ampls = np.array(ampls) / nf

        ams.append(ampls)

    ampls = np.mean(np.array(ams), axis=0)

    if 'plot_filename' in sweep_cfg:
        plot_frequency_response(freqs, ampls, sweep_cfg['plot_filename'])

    if 'csv_filename' in sweep_cfg:
        write_csv(sweep_cfg['csv_filename'], freqs, ampls)

if __name__ == '__main__':
    main()