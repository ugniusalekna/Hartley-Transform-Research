import os
import wave
from scipy.io import wavfile

from functions.signal_manipulation import *
from functions.transforms import *
from functions.frequency_manipulation import *
from functions.plotting import *
from functions.hartley_manipulation import * 

os.chdir(os.path.dirname(os.path.abspath(__file__)))
    

def display_hartley_spectrum(args):

    sample_rate, signal_data, wav_info = args

    signal_fht = apply_dht(signal_data)
    plot_hartley_spectrum(sample_rate, signal_data)
    truncated_amplitude, ratios = split_hartley_transform(signal_fht)

    plot_signal(sample_rate, signal_data, truncated_amplitude, wav_info, title=f"Spectral Analysis of file '{file_name}'")
    # animate_hartley_spectrum(sample_rate, signal_data.reshape(-1))
    

def apply_processing(args):
    
    sample_rate, signal_data, wav_info = args

    methods = ['Band-Reject Filter', 'Threshold Filter', 'Add Frequency', 'Scale Amplitude', 'Shift Frequencies']
    cutoff_lower=100
    cutoff_upper=5000
    threshold=1
    freq_to_add=250
    amplitude_to_add=30
    freq_to_scale=20
    scale_factor=2
    shift=250

    for processing_method in methods:

        modified_signal = modify_amplitude(sample_rate, signal_data, processing_method,
                                                cutoff_lower, cutoff_upper, threshold, freq_to_add,
                                                amplitude_to_add, freq_to_scale, scale_factor, shift)


        signal_fht_modified = apply_dht(modified_signal, matrix_form=True)
        reconstructed_signal = inverse_dht(signal_fht_modified)
        truncated_amplitude_modified, ratios = split_hartley_transform(signal_fht_modified)

        plot_signal(sample_rate, modified_signal, truncated_amplitude_modified, wav_info, title=f"'{processing_method}' Applied to File {file_name}")

        plot_hartley_spectrum(sample_rate, reconstructed_signal)


def compare_runtime():

    file_path = '..\\data\\Sounds\\Dirbtiniai\\2500Hz_sine.wav'
    sample_rate, signal_data = wavfile.read(file_path, 'r')

    print(f' - - - - Running Time Comparison - - - - ')
    print(f'Signal length: {sample_rate * 1}')
    apply_dht(signal_data)
    apply_dft(signal_data)
    apply_fht(signal_data)
    

if __name__ == '__main__':
    # file_path = '..\\data\\Sounds\\Dirbtiniai\\10Hz_30Hz_sine.wav'
    # file_path = '..\\data\\Sounds\\Dirbtiniai\\noise_50Hz_sine.wav'
    file_path = '..\\data\\Sounds\\Dirbtiniai\\50Hz_100Hz_150Hz_200Hz_250Hz_sine.wav'
    
    sample_rate, signal_data = wavfile.read(file_path, 'r')
    wave_object = wave.open(file_path, mode='r')
    num_of_channels = wave_object.getnchannels()
    bit_depth = wave_object.getsampwidth() * 8
    wave_object.close()
    
    signal_data = signal_data.reshape(-1, num_of_channels) # Reshape to a column vector
    signal_data = normalize(signal_data) # Normalize to [-1, 1]
    signal_data = extract_segment(sample_rate, signal_data, start_time=0, end_time=0.2, time_unit='s')

    file_name = os.path.basename(file_path)
    wav_info = [file_name, num_of_channels, bit_depth]
    args = [sample_rate, signal_data, wav_info]
    display_hartley_spectrum(args)
    apply_processing(args)
    compare_runtime()