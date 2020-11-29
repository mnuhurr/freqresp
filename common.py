
import yaml

import sounddevice as sd
import numpy as np

import matplotlib.pyplot as plt

import csv


def load_config(filename='settings.yaml'):
    '''
    read a yaml file

    :param filename: filename
    :return: dict
    '''
    cfg = {}

    with open(filename, 'rt') as f:
        cfg = yaml.safe_load(f)

    return cfg


def find_device_id(dev_str):
    '''
    find audio device id by looking for a substring in the device name

    :param dev_str: substring to search in the device name
    :return: device id. returns None if no device is found
    '''

    device_list = sd.query_devices()

    for dev_ind, dev in enumerate(device_list):
        if dev_str in dev['name']:
            return dev_ind

    return None


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


def plot_frequency_response(fr, amp, fn='fr.png'):
    fig = plt.figure(figsize=(14,4))
    plt.semilogx(fr, 20 * np.log10(amp))
    plt.xlabel('frequency')
    plt.ylabel('amplitude')
    plt.grid(True)
    plt.savefig(fn)

def write_csv(csv_fn, freqs, ampls, use_db=True):
    '''
    write frequency response to a csv file

    :param csv_fn: filename
    :param freqs: frequencies
    :param ampls: amplitudes
    :param use_db: convert amplitudes to decibels
    :return: None
    '''

    with open(csv_fn, 'wt') as f:
        csv_writer = csv.writer(f)
        hdr = ['f', 'a']

        csv_writer.writerow(hdr)

        for row in zip(freqs, ampls):
            if use_db:
                row[1] = 20 * np.log10(row[1])

            csv_writer.writerow(row)



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

