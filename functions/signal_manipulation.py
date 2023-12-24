import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_time_scale(duration_sec):
    # Determine the appropriate time scale based on duration
    if duration_sec < 1:
        time_scale = 1000
        xlabel = 'ms'
    elif duration_sec < 60:
        time_scale = 1
        xlabel = 's'
    elif duration_sec < 60**2:
        time_scale = 1/60
        xlabel = 'min'
    else:
        time_scale = 1/60**2
        xlabel = 'h'

    return time_scale, xlabel


def get_duration_sec(sample_rate, signal_data):
    # Calculate signal duration in seconds
    return signal_data.shape[0] / sample_rate


def normalize(signal_data):
    # Return signal amplitude values normalized to [-1, 1] 
    return signal_data/np.max(np.abs(signal_data))


def get_time_axis(sample_rate, signal_data, time_scale):
    # Calculate time linspace of the given signal
    return np.arange(0, signal_data.shape[0]) / sample_rate * time_scale


def extract_segment(sample_rate, signal_data, start_time, end_time, time_unit):
    # Slice signal array to extract only a section of it
    start_index = np.round(convert_to_seconds(start_time, time_unit) * sample_rate).astype(int)
    end_index = np.round(convert_to_seconds(end_time, time_unit) * sample_rate).astype(int)
    segment = signal_data[start_index:end_index, :]
    
    return segment


def convert_to_seconds(numeric_value, time_unit):
    # Convert user input to seconds for further use
    match time_unit:
        case 'ms':
            return numeric_value / 1000
        case 's':
            return numeric_value
        case 'min':
            return numeric_value * 60
        case 'h':
            return numeric_value * 60**2
        

def apply_window(sample_rate, signal_data, num_channels, window_function):
    # Obtain time interval values
    duration = get_duration_sec(sample_rate, signal_data)
    time_scale, _ = get_time_scale(duration)
    time = get_time_axis(sample_rate, signal_data, time_scale)

    # Get values of window function in time interval
    match window_function:
        case 'hamming':
            window_function = np.hamming(len(time))
        case 'hanning':
            window_function = np.hanning(len(time))
        case 'blackman':
            window_function = np.blackman(len(time))
        case 'parzen':
            window_function = np.parzen(len(time))
        case 'bartlett':
            window_function = np.bartlett(len(time))

    windowed_audio = np.zeros_like(signal_data)
    for i in range(num_channels):
        windowed_audio[:, i] = window_function * signal_data[:, i]

    return windowed_audio