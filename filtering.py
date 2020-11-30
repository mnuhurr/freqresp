import numpy as np
from scipy.signal import butter, lfilter

# debugs
import matplotlib.pyplot as plt
from scipy.signal import freqz

def butter_bandpass(f, sr, width=500, order=3):
    '''

    :param sgn:
    :param f:
    :param sr:
    :return:
    '''

    f_lc = f - width / 2
    f_hc = f + width / 2

    f_lc = np.max([0.5, f_lc])
    f_hc = np.min([sr / 2 - 0.5, f_hc])

    f_nyq = sr / 2
    lc = f_lc / f_nyq
    hc = f_hc / f_nyq

    b, a = butter(order, [lc, hc], btype='band')

    return b, a


def butter_lowpass(f, sr, order=3):

    f_c = f / (sr/2)

    b, a = butter(order, f_c, btype='lowpass')

    return b, a

def bandpass(sgn, f, sr, order=3):
    b, a = butter_bandpass(f, sr, width=200, order=order)
    return lfilter(b, a, sgn)


def lowpass(sgn, f, sr, order=5):
    b, a = butter_lowpass(f, sr, order=order)
    return lfilter(b, a, sgn)

def plot_filter(b, a, sr, filename):
    plt.figure(figsize=(8,8))
    w, h = freqz(b, a, worN=2000)

    plt.semilogx(sr / (2*np.pi) * w, 20 * np.log10(np.abs(h)))
    plt.grid(True)
    plt.savefig(filename)

if __name__ == '__main__':
    f = 10000
    sr = 192000

    b, a = butter_bandpass(f, sr, width=200, order=3)
    #b, a = butter_lowpass(f, sr, order=9)
    plot_filter(b, a, sr, 'filsu.png')

