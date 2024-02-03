import numpy as np
import os
import matplotlib.pyplot as plt

from functions.signal_manipulation import *
from functions.transforms import *
from functions.frequency_manipulation import *
from functions.plotting import *

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_freq_axis(sample_rate, signal_amplitude):
    return np.linspace(0, sample_rate/2, signal_amplitude.shape[0])


def get_nonzero_amplitudes(signal_dht, tol=1e-7):

    idx = np.where(np.abs(signal_dht) > tol)[0]
    nonzero_amplitudes = signal_dht[idx]
    return idx, nonzero_amplitudes


def get_ratios(signal_data):

    signal_dht = apply_dht(signal_data)
    _, amplitudes = get_nonzero_amplitudes(signal_dht, tol=1e-3)

    n = len(amplitudes)
    ratios = []

    for i in range(n // 2): 
        denominator = amplitudes[-(i + 1)]
        numerator = amplitudes[i]

        if denominator == 0 and numerator == 0:
            ratio = 1
            print('Denominator and numerator are 0')

        elif denominator == 0:
            ratio = np.inf
        else:
            ratio = numerator / denominator
        ratios.append(ratio)

    return np.round(ratios, 2)


def print_nonzero_amplitudes(sample_rate, signal_data):
    signal_dht = apply_dht(signal_data)
    f = get_freq_axis(2*sample_rate, signal_dht)
    idx, nonzero_amplitudes = get_nonzero_amplitudes(signal_dht, tol=1e-3)
    for i, amp in zip(idx, nonzero_amplitudes):
        print(f'Amplitude at {int(np.round(f[i]))} Hz: {amp}')


def custom_division(numerator, denominator):
    if np.isinf(numerator) and np.isinf(denominator):
        return np.sign(numerator) * np.sign(denominator)
    elif np.isinf(denominator):
        return 0.0
    elif numerator == 0 and denominator == 0:
        return 1.0
    elif denominator == 0:
        return np.sign(numerator) * np.inf

    sign = ''
    if (numerator > 0 and denominator < 0) or (numerator < 0 and denominator > 0):
        sign = '+-' if numerator > 0 else '-+'

    return f"{sign}{np.abs(numerator / denominator)}" if sign else numerator / denominator


def split_hartley_transform(dht):

    non_zero_freq_dht = dht[1:]
    
    if len(non_zero_freq_dht) % 2 == 0:
        mid = len(non_zero_freq_dht) // 2
        left_half = non_zero_freq_dht[:mid]
        right_half = non_zero_freq_dht[mid:]
    else:
        mid = len(non_zero_freq_dht) // 2
        left_half = non_zero_freq_dht[:mid]
        right_half = non_zero_freq_dht[mid + 1:]
        mid_value = non_zero_freq_dht[mid]
    
    right_half = np.flip(right_half)
    ratios = [custom_division(left_half[i], right_half[i]) for i in range(len(left_half))]

    left_half_with_zero = np.insert(2*left_half, 0, np.abs(dht[0]))
    if len(non_zero_freq_dht) % 2 != 0:
        left_half_with_zero = np.append(left_half_with_zero, mid_value)

    return np.abs(left_half_with_zero), ratios


def reconstruct_hartley_spectrum(left_half_with_zero, ratios):

    def does_not_start_with_plus_minus(string):
        return not (string.startswith('+-') or string.startswith('-+'))

    if len(left_half_with_zero) - len(ratios) == 2:
        mid_value = left_half_with_zero[-1]
        N = 2 * (len(left_half_with_zero) - 1)
        reconstructed = np.zeros(N)
        left_half = left_half_with_zero[1:-1]/2
    else:
        N = 2 * len(left_half_with_zero) - 1
        reconstructed = np.zeros(N)
        left_half = left_half_with_zero[1:]/2

    right_half = []
    for i, ratio in enumerate(ratios):
        if isinstance(ratio, str):
            if does_not_start_with_plus_minus(ratio):
                value = left_half[i] / float(ratio)
            else: 
                sign = ratio[:2]
                num_ratio = float(ratio[2:]) if ratio[2:] != 'inf' else np.inf
                value = left_half[i] / num_ratio
                left_half[i] = -left_half[i] if sign == '-+' else left_half[i]
                value = -value if sign == '+-' else value
        else:
            value = left_half[i] / ratio
        right_half.append(value)

    reconstructed[0] = left_half_with_zero[0]

    reconstructed[1:len(left_half) + 1] = left_half
    if len(left_half_with_zero) - len(ratios) == 2:
        reconstructed[len(left_half) + 1] = mid_value
    reconstructed[-len(right_half):] = np.flip(right_half)

    return reconstructed