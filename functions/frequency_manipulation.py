import numpy as np
import os

from functions.signal_manipulation import *
from functions.transforms import *
from functions.hartley_manipulation import * 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_freq_axis(sample_rate, signal_amplitude):
    # Calculate the linspace of frequency values
    return np.linspace(0, sample_rate/2, signal_amplitude.shape[0])

def truncate_amplitude(signal_data):
    
    amplitude, _ = apply_fft(signal_data)
    num_channels = amplitude.shape[1]
    N = amplitude.shape[0]
    # Truncate half of the symmetric amplitude spectrum
    truncated = np.zeros((N//2 + 1 if N % 2 == 0 else (N+1)//2, num_channels))
    for i in range(num_channels):
        if N % 2 == 0:
            truncated_channel = amplitude[:N//2 + 1, i].copy()
            truncated_channel[1:-1] *= 2
        else:
            truncated_channel = amplitude[:(N+1)//2, i].copy()
            truncated_channel[1:] *= 2
        truncated[:, i] = truncated_channel
    
    return truncated

def reconstruct_amplitude(truncated, N):
    # Reconstruct the symmetry of the amplitude spectrum
    num_channels = truncated.shape[1]
    reconstructed = np.zeros((N, num_channels))
    for i in range(num_channels):
        if N % 2 == 0:
            reconstructed[1:N//2, i] = truncated[1:-1, i] / 2
            reconstructed[N//2, i] = truncated[-1, i]
            reconstructed[N//2 + 1:, i] = truncated[1:-1, i][::-1] / 2
        else:
            reconstructed[1:(N+1)//2, i] = truncated[1:, i] / 2
            reconstructed[(N+1)//2:, i] = truncated[1:, i][::-1] / 2
        reconstructed[0, i] = truncated[0, i]  # DC component

    return reconstructed


def modify_amplitude(sample_rate, signal_data, method,
                     cutoff_lower=None, cutoff_upper=None, threshold=None, freq_to_add=None,
                     amplitude_to_add=None, freq_to_scale=None, scale_factor=None, shift=None):
    # Apply frequency domain processing
    match method:
        case 'Band-Reject Filter':
            modified_signal = band_reject_filter(sample_rate, signal_data, cutoff_lower, cutoff_upper)
        case 'Threshold Filter':
            modified_signal = threshold_filter(sample_rate, signal_data, threshold)
        case 'Add Frequency':
            modified_signal = add_frequency(sample_rate, signal_data, freq_to_add, amplitude_to_add)    
        case 'Scale Amplitude':
            modified_signal = scale_amplitude(sample_rate, signal_data, freq_to_scale, scale_factor)
        case 'Shift Frequencies':
            modified_signal = shift_frequencies(sample_rate, signal_data, shift)

    return modified_signal


def band_reject_filter(sample_rate, signal_data, cutoff_lower, cutoff_upper):
 
    signal_fht = apply_dht(signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht.reshape(-1))
    truncated_amplitude = truncated_amplitude.reshape(-1, 1)
    # truncated_amplitude = truncate_amplitude(signal_data)
    freq_axis = get_freq_axis(sample_rate, truncated_amplitude)
    filtered_amplitude = np.copy(truncated_amplitude)
    # Apply band-reject filter
    if cutoff_lower is not None and cutoff_upper is not None:
        mask = (freq_axis >= cutoff_lower) & (freq_axis <= cutoff_upper)
    elif cutoff_lower is not None:
        mask = freq_axis >= cutoff_lower
    elif cutoff_upper is not None:
        mask = freq_axis <= cutoff_upper
    filtered_amplitude[mask, :] = 0

    # reconstructed_amplitude = reconstruct_amplitude(filtered_amplitude, amplitude.shape[0])
    filtered_amplitude = filtered_amplitude.reshape(-1)
    reconstructed_fht = reconstruct_hartley_spectrum(filtered_amplitude, ratios)
    reconstructed_signal = inverse_dht(reconstructed_fht)

    return reconstructed_signal


def threshold_filter(sample_rate, signal_data, threshold):
 
    # amplitude, phase = apply_fft(signal_data)
    # truncated_amplitude = truncate_amplitude(signal_data)
    signal_fht = apply_dht(signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht.reshape(-1))
    truncated_amplitude = truncated_amplitude.reshape(-1, 1)
    filtered_amplitude = np.copy(truncated_amplitude)
    num_channels = signal_data.shape[1]
    # Apply threshold filter
    for i in range(num_channels):
        mask = (truncated_amplitude[:, i] <= threshold)
        filtered_amplitude[mask, i] = 0

    # reconstructed_amplitude = reconstruct_amplitude(filtered_amplitude, amplitude.shape[0])
    # reconstructed_signal = inverse_fft(reconstructed_amplitude, phase)
    filtered_amplitude = filtered_amplitude.reshape(-1)
    reconstructed_fht = reconstruct_hartley_spectrum(filtered_amplitude, ratios)
    reconstructed_signal = inverse_dht(reconstructed_fht)

    return reconstructed_signal


def add_frequency(sample_rate, signal_data, freq_to_add, amplitude_to_add):

    # amplitude, phase = apply_fft(signal_data)
    # truncated_amplitude = truncate_amplitude(signal_data)
    signal_fht = apply_dht(signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht.reshape(-1))
    truncated_amplitude = truncated_amplitude.reshape(-1, 1)
    freq_axis = get_freq_axis(sample_rate, truncated_amplitude)    
    added_amplitude = np.copy(truncated_amplitude)
    # Add frequency
    idx = np.where((freq_axis > freq_to_add - 1) & (freq_axis < freq_to_add + 1))[0]
    added_amplitude[idx, :] += amplitude_to_add
    # reconstructed_amplitude = reconstruct_amplitude(added_amplitude, amplitude.shape[0])
    # reconstructed_signal = inverse_fft(reconstructed_amplitude, phase)
    added_amplitude = added_amplitude.reshape(-1)
    reconstructed_fht = reconstruct_hartley_spectrum(added_amplitude, ratios)
    reconstructed_signal = inverse_dht(reconstructed_fht)


    return reconstructed_signal


def scale_amplitude(sample_rate, signal_data, freq_to_scale, scale_factor):
    
    # amplitude, phase = apply_fft(signal_data)
    # truncated_amplitude = truncate_amplitude(signal_data)
    signal_fht = apply_dht(signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht.reshape(-1))
    truncated_amplitude = truncated_amplitude.reshape(-1, 1)
    freq_axis = get_freq_axis(sample_rate, truncated_amplitude)    
    scaled_amplitude = np.copy(truncated_amplitude)
    # Scale frequency
    idx = np.where((freq_axis > freq_to_scale - 1) & (freq_axis < freq_to_scale + 1))[0]
    scaled_amplitude[idx, :] *= scale_factor
    # reconstructed_amplitude = reconstruct_amplitude(scaled_amplitude, amplitude.shape[0])
    # reconstructed_signal = inverse_fft(reconstructed_amplitude, phase)
    scaled_amplitude = scaled_amplitude.reshape(-1)
    reconstructed_fht = reconstruct_hartley_spectrum(scaled_amplitude, ratios)
    reconstructed_signal = inverse_dht(reconstructed_fht)

    return reconstructed_signal


def shift_frequencies(sample_rate, signal_data, shift):
    
    # amplitude, phase = apply_fft(signal_data)
    # truncated_amplitude = truncate_amplitude(signal_data)
    signal_fht = apply_dht(signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht.reshape(-1))
    truncated_amplitude = truncated_amplitude.reshape(-1, 1)
    # Convert Hz shift to sample shift
    shift_samples = np.round(shift / sample_rate * signal_fht.shape[0]).astype(int)
    # Shift the frequency spectrum excluding DC component
    shifted_amplitude = np.concatenate([[truncated_amplitude[0, :]], np.roll(truncated_amplitude[1:, :], shift_samples)])

    # reconstructed_amplitude = reconstruct_amplitude(shifted_amplitude, amplitude.shape[0])
    # reconstructed_signal = inverse_fft(reconstructed_amplitude, phase)
    shifted_amplitude = shifted_amplitude.reshape(-1)
    reconstructed_fht = reconstruct_hartley_spectrum(shifted_amplitude, ratios)
    reconstructed_signal = inverse_dht(reconstructed_fht)

    return reconstructed_signal