import numpy as np
import os
import wave
import matplotlib.pyplot as plt
from scipy.io import wavfile
import tkinter as tk
from tkinter import filedialog, ttk

from functions.signal_manipulation import *
from functions.transforms import *
from functions.frequency_manipulation import *
from functions.plotting import *
from functions.hartley_manipulation import * 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# file_path = 'Sounds\\Dirbtiniai\\10Hz_30Hz_sine.wav'
# file_path = 'Sounds\\Dirbtiniai\\noise_50Hz_sine.wav'
file_path = 'Sounds\\Dirbtiniai\\50Hz_100Hz_150Hz_200Hz_250Hz_sine.wav'
sample_rate, signal_data = wavfile.read(file_path, 'r')



wave_object = wave.open(file_path, mode='r')
num_of_channels = wave_object.getnchannels()
bit_depth = wave_object.getsampwidth() * 8
wave_object.close()
file_name = os.path.basename(file_path)
wav_info = [file_name, num_of_channels, bit_depth]

signal_data = signal_data.reshape(-1, num_of_channels) # Reshape to a column vector
signal_data = normalize(signal_data) # Normalize to [-1, 1]

# Ask the user for start time (end time is fixed to 200 ms)
start_time, end_time, time_unit = 0, 0.2, 's'
signal_data = extract_segment(sample_rate, signal_data, start_time, end_time, time_unit)


signal_fht = apply_dht(signal_data)
plot_hartley_spectrum(sample_rate, signal_data)
truncated_amplitude, ratios = split_hartley_transform(signal_fht)

plot_signal(sample_rate, signal_data, truncated_amplitude, wav_info, title=f"Spectral Analysis of file '{file_name}'")

methods = ['Band-Reject Filter', 'Threshold Filter', 'Add Frequency', 'Scale Amplitude', 'Shift Frequencies']

processing_method = methods[4]
cutoff_lower=100
cutoff_upper=5000
threshold=1
freq_to_add=250
amplitude_to_add=30
freq_to_scale=20
scale_factor=2
shift=250


modified_signal = modify_amplitude(sample_rate, signal_data, processing_method,
                                           cutoff_lower, cutoff_upper, threshold, freq_to_add,
                                           amplitude_to_add, freq_to_scale, scale_factor, shift)


signal_fht_modified = apply_dht(modified_signal, matrix_form=True)
reconstructed_signal = inverse_dht(signal_fht_modified)
truncated_amplitude_modified, ratios = split_hartley_transform(signal_fht_modified)

plot_signal(sample_rate, modified_signal, truncated_amplitude_modified, wav_info, title=f"Spectral Analysis of file '{file_name}'")

plot_hartley_spectrum(sample_rate, reconstructed_signal)















# test_data = np.array([0, -20, 6, 0, 0, 0, -3, 10])

# print(test_data)
# left, ratios = split_hartley_transform(test_data)
# print(left, ratios)








# ani = animate_hartley_spectrum(sample_rate, signal_data.reshape(-1))













# file_path = 'Sounds\\Dirbtiniai\\5000Hz_sine.wav'
# sample_rate, signal_data = wavfile.read(file_path, 'r')

# print(f'Signalo reikšmių ilgis: {sample_rate * 1}')
# # apply_dht(signal_data)
# # apply_dft(signal_data)
# # apply_matrix_dht(signal_data)
# apply_fht(signal_data)

