import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import tkinter as tk
from tkinter import filedialog, ttk
import winsound
import os

from functions.signal_manipulation import *
from functions.transforms import *
from functions.frequency_manipulation import *
from functions.plotting import *
from functions.hartley_manipulation import * 


os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Create main GUI window
root = tk.Tk()
root.title('WAV File Reader')
root.resizable(False, False)
root.geometry('300x125')
def on_root_close():
    # Define root closing protocol
    plt.close('all')
    if os.path.exists('temp'):
        for filename in os.listdir('temp'):
            file_path = os.path.join('temp', filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir('temp') 
    for window in root.winfo_children():
        if isinstance(window, tk.Toplevel):
            window.destroy()
    root.destroy()
root.protocol('WM_DELETE_WINDOW', on_root_close)    


def open_file():
    # Create dialog window to select wav files 
    file_path = filedialog.askopenfilename(
        filetypes=(
            ("wav files", "*.wav"),
            ("all files", "*.*")))
    if file_path:
        # Create folder for temp files
        if not os.path.exists('temp'):
            os.mkdir('temp')
        
        # Read WAV file and get attributes
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
        start_time, end_time, time_unit = ask_segment_time_input(file_path, sample_rate, signal_data)
        signal_data = extract_segment(sample_rate, signal_data, start_time, end_time, time_unit)

        # Truncate amplitude spectrum for more accurate visualization
        signal_fht = apply_dht(signal_data)
        truncated_amplitude, ratios = split_hartley_transform(signal_fht)
        truncated_amplitude = truncated_amplitude.reshape(-1, 1)
        # truncated_amplitude = truncate_amplitude(windowed_signal) # Truncated amplitude changed to left side of magnitude spectrum of fht

        # Plot every channel of a signal and their amplitude spectra
        plot_signal(sample_rate, signal_data, truncated_amplitude, wav_info, title=f"Spectral Analysis of file '{file_name}'")

        # Display signal processing window
        process_signal_options(sample_rate, signal_data, wav_info)


def play_audio(file_path):
    winsound.PlaySound(file_path, winsound.SND_ASYNC + winsound.SND_LOOP)

def pause_audio():
    winsound.PlaySound(None, 0)

def write_file(sample_rate, signal_data, output_filename):
    # Write the NumPy array as a WAV file
    output_file = 'temp\\' + output_filename + '.wav'  # Output WAV file name
    wavfile.write(output_file, sample_rate, signal_data.astype(np.float32))
    # Get full path to the written temporary audio file
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = script_path + '\\' + output_file

    return file_path


def ask_segment_time_input(file_path, sample_rate, signal_data):

    duration_sec = get_duration_sec(sample_rate, signal_data)
    time_scale, xlabel = get_time_scale(duration_sec)
    duration_scaled = duration_sec * time_scale

    # Time input window
    segment_window = tk.Toplevel(root)
    segment_window.title("Time Input")
    segment_window.geometry('300x175')
    segment_window.resizable(False, False)
    def on_closing():
        pause_audio()  # Stop audio when the window is closed
        segment_window.destroy()
    segment_window.protocol("WM_DELETE_WINDOW", on_closing)  # Intercept window closing event

    start_time = tk.StringVar()
    time_unit = tk.StringVar()

    ttk.Label(segment_window, text=f"File: {os.path.basename(file_path)}").grid(row=0, column=0, columnspan=4, padx=10, pady=5)
    ttk.Label(segment_window, text=f"Duration: {duration_scaled:.2f} {xlabel}").grid(row=1, column=0, columnspan=4, padx=10, pady=5)

    # Labels and Entries for the widgets
    ttk.Label(segment_window, text='Start Time:').grid(row=2, column=0, padx=10, pady=5, sticky='e')
    ttk.Entry(segment_window, textvariable=start_time, width=6).grid(row=2, column=1, padx=5, pady=5, sticky='w')
    tk.Button(segment_window, text="Play Audio", command=lambda: play_audio(file_path), height=1, width=10).grid(row=2, column=2, padx=10, pady=5)
    ttk.Label(segment_window, text='Time Unit:').grid(row=3, column=0, padx=10, pady=5, sticky='e')
    ttk.Combobox(segment_window, textvariable=time_unit, state='readonly', values=('ms', 's', 'min', 'h'), width=4).grid(row=3, column=1, padx=5, pady=5, sticky='w')
    tk.Button(segment_window, text="Pause Audio", command=lambda: pause_audio(), height=1, width=10).grid(row=3, column=2, padx=10, pady=5)
    tk.Button(segment_window, text='OK', command=on_closing, height=1, width=10).grid(row=4, column=2, padx=10, pady=5)

    # Configure columns to center the displayed widgets 
    segment_window.grid_columnconfigure(0, weight=1)
    segment_window.grid_columnconfigure(1, weight=1)
    segment_window.grid_columnconfigure(2, weight=1)

    # Wait for the dialog window to close
    segment_window.wait_window()

    if start_time.get() and time_unit.get():
        # Implement checking if start_time + 200 ms < signal_duration
        if float(start_time.get()) + 0.2 * time_scale < duration_scaled:
            return float(start_time.get()), float(start_time.get()) + 0.2 * time_scale, time_unit.get()
    else:
        print('Start time value not selected. Returning first 200 ms of the signal.')
        return 0, 200, 'ms'


def process_signal_options(sample_rate, signal_data, wav_info):

    def update_processing_options(*args):
        # Refresh window with new widgets
        for widget in option_widgets:
            widget.grid_forget()

        # Display widgets for each processing method
        processing_method = processing_method_var.get()
        match processing_method:
            case 'Band-Reject Filter':
                cutoff_lower_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
                cutoff_lower_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
                cutoff_upper_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
                cutoff_upper_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
            case 'Threshold Filter':
                threshold_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
                threshold_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
            case 'Add Frequency':
                freq_to_add_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
                freq_to_add_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
                amplitude_to_add_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
                amplitude_to_add_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
            case 'Scale Amplitude':
                freq_to_scale_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
                freq_to_scale_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
                scale_factor_label.grid(row=3, column=0, padx=10, pady=5, sticky='e')
                scale_factor_entry.grid(row=3, column=1, padx=5, pady=5, sticky='w')
            case 'Shift Frequencies':
                shift_label.grid(row=2, column=0, padx=10, pady=5, sticky='e')
                shift_entry.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        apply_button.grid(row=4, column=1, padx=10, pady=15)

    def apply_processing():
        
        processing_method = processing_method_var.get()
        cutoff_lower = cutoff_lower_var.get()
        cutoff_upper = cutoff_upper_var.get()
        threshold = threshold_var.get()
        freq_to_add = freq_to_add_var.get()
        amplitude_to_add = amplitude_to_add_var.get()
        freq_to_scale = freq_to_scale_var.get()
        scale_factor = scale_factor_var.get()
        shift = shift_var.get()

        # Do original signal's frequency processing (for playing)
        modified_signal = modify_amplitude(sample_rate, signal_data, processing_method,
                                           cutoff_lower, cutoff_upper, threshold, freq_to_add,
                                           amplitude_to_add, freq_to_scale, scale_factor, shift)
        # Truncate half of the amplitude (for displaying)
        truncated_amplitude_modified = split_hartley_transform(apply_dht(modified_signal))
        # Write and play temp files (for playing)
        file_path_original = write_file(sample_rate, signal_data, output_filename="original_audio")
        file_path_processed = write_file(sample_rate, modified_signal, output_filename="processed_audio")
        play_audio_files(wav_info[0], file_path_original, file_path_processed)

        # Plot processed signal and its amplitude spectrum
        plot_signal(sample_rate, modified_signal, truncated_amplitude_modified, wav_info, title=f"'{processing_method}' Applied to File {wav_info[0]}")

    # Processing selection window
    processing_window = tk.Toplevel(root)
    processing_window.title('Processing Options')
    processing_window.geometry('300x175')
    processing_window.resizable(False, False)

    # Labels and Entries for the widgets
    processing_method_var = tk.StringVar()
    processing_method_var.trace('w', update_processing_options) # Trace variable to update window with new widgets
    processing_method_label = ttk.Label(processing_window, text='Processing Method:')
    processing_method_dropdown = ttk.Combobox(processing_window, textvariable=processing_method_var, state='readonly',
                                              values=['Band-Reject Filter', 'Threshold Filter', 'Add Frequency', 'Scale Amplitude', 'Shift Frequencies'],
                                              width=17)
    processing_method_label.grid(row=0, column=0, padx=10, pady=15, sticky='e')
    processing_method_dropdown.grid(row=0, column=1, padx=10, pady=5, sticky='w')

    cutoff_lower_var = tk.DoubleVar()
    cutoff_upper_var = tk.DoubleVar()
    threshold_var = tk.DoubleVar()
    freq_to_add_var = tk.DoubleVar()
    amplitude_to_add_var = tk.DoubleVar()
    freq_to_scale_var = tk.DoubleVar()
    scale_factor_var = tk.DoubleVar()
    shift_var = tk.IntVar()

    cutoff_lower_label = ttk.Label(processing_window, text='Cutoff Lower (Hz):')
    cutoff_lower_entry = ttk.Entry(processing_window, textvariable=cutoff_lower_var, width=6)
    cutoff_upper_label = ttk.Label(processing_window, text='Cutoff Upper (Hz):')
    cutoff_upper_entry = ttk.Entry(processing_window, textvariable=cutoff_upper_var, width=6)
    threshold_label = ttk.Label(processing_window, text='Threshold:')
    threshold_entry = ttk.Entry(processing_window, textvariable=threshold_var, width=6)
    freq_to_add_label = ttk.Label(processing_window, text='Frequency to Add (Hz):')
    freq_to_add_entry = ttk.Entry(processing_window, textvariable=freq_to_add_var, width=6)
    amplitude_to_add_label = ttk.Label(processing_window, text='Amplitude to Add:')
    amplitude_to_add_entry = ttk.Entry(processing_window, textvariable=amplitude_to_add_var, width=6)
    freq_to_scale_label = ttk.Label(processing_window, text='Frequency to Scale (Hz):')
    freq_to_scale_entry = ttk.Entry(processing_window, textvariable=freq_to_scale_var, width=6)
    scale_factor_label = ttk.Label(processing_window, text='Scale Factor:')
    scale_factor_entry = ttk.Entry(processing_window, textvariable=scale_factor_var, width=6)
    shift_label = ttk.Label(processing_window, text='Shift (Hz):')
    shift_entry = ttk.Entry(processing_window, textvariable=shift_var, width=6)

    option_widgets = [cutoff_lower_label, cutoff_lower_entry, cutoff_upper_label, cutoff_upper_entry, threshold_label,
                      threshold_entry, freq_to_add_label, freq_to_add_entry, amplitude_to_add_label, amplitude_to_add_entry,
                      freq_to_scale_label, freq_to_scale_entry, scale_factor_label, scale_factor_entry, shift_label, shift_entry]
    
    apply_button = tk.Button(processing_window, text='Apply', command=apply_processing, height=1, width=10)

    processing_window.mainloop()


def play_audio_files(file_name, path_original, path_processed):
    # Audio player window
    audio_window = tk.Toplevel(root)
    audio_window.title("Audio Player")
    audio_window.geometry('300x150')
    def on_closing():
        pause_audio()  # Stop audio when the window is closed
        file_paths = [path_original, path_processed]
        for file_path in file_paths:
            if os.path.isfile(file_path):
                os.remove(file_path)
        plt.close(plt.gcf())
        audio_window.destroy()
    audio_window.protocol("WM_DELETE_WINDOW", on_closing)  # Intercept window closing event

    # Display the audio file name
    audio_filename_label = tk.Label(audio_window, text=f"Selected Audio: {file_name}")
    audio_filename_label.pack()
    audio_filename_label.configure(anchor="center")

    # Display the duration of the audio
    signal_duration_label = tk.Label(audio_window, text=f"Signal Duration: 200 ms")
    signal_duration_label.pack()
    
    # Create widgets for playing/pausing original and processed audio
    play_original_button = tk.Button(audio_window, text="Play Original Audio", command=lambda: play_audio(path_original), width=20)
    play_original_button.pack(pady=(5, 0), padx=10)
    modulate_button = tk.Button(audio_window, text="Play Processed Audio", command=lambda: play_audio(path_processed), width=20)
    modulate_button.pack(pady=5, padx=10)
    pause_button = tk.Button(audio_window, text="Pause", command=lambda: pause_audio(), width=20)
    pause_button.pack(pady=(0, 5), padx=10)


# Create a button widget that triggers 'open_file'
open_button = tk.Button(root, text='Read WAV File', command=open_file, height=2, width=15)
open_button.pack(expand=True, pady=20)
# Start the tkinter mainloop, which keeps GUI responsive
root.mainloop()