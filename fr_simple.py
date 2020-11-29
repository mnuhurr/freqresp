'''

simple tool to measure frequency response. generates several sinewaves and measures the maximum amplitude of the
response. more sophisticated methods to come.

'''

import numpy as np
import sounddevice as sd

from tqdm import tqdm

from common import load_config, find_device_id, generate_sine, crop_signal, generate_frequency_range
from common import plot_frequency_response, write_csv



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