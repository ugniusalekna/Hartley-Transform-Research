import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.animation import FuncAnimation, PillowWriter  # Import PillowWriter for GIF

from matplotlib.lines import Line2D
import os

from functions.signal_manipulation import *
from functions.frequency_manipulation import *
from functions.transforms import *
from functions.hartley_manipulation import * 

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def get_freq_axis(sample_rate, signal_amplitude):
    # Calculate the linspace of frequency values
    return np.linspace(0, sample_rate/2, len(signal_amplitude))


def plot_signal(sample_rate, signal_data, signal_amplitude, wav_info, title):
    # Obtain and calculate relevant data 
    duration_sec = get_duration_sec(sample_rate, signal_data)
    time_scale, xlabel = get_time_scale(duration_sec)
    duration_scaled = duration_sec * time_scale
    num_channels = wav_info[1]
    text_info = f'Channels: {wav_info[1]}, Audio Duration: {np.round(duration_scaled, 2)} {xlabel}, Sampling Rate: {sample_rate} Hz, Bit Depth: {wav_info[2]} bits'
    
    # Create a figure with multiple subplots based on the number of channels
    fig, axs = plt.subplots(2, num_channels, figsize=(12, 9))
    axs = axs.reshape(-1, num_channels)
    time_axis_signal = get_time_axis(sample_rate, signal_data, time_scale)
    freq_axis_signal = get_freq_axis(sample_rate, signal_amplitude)

    for channel in range(num_channels):        
        # Plot the signal and its amplitude spectrum for every channel
        axs[0, channel].plot(time_axis_signal, signal_data, label='Original signal', linewidth=0.4)
        if num_channels > 1:
            axs[0, channel].set_title(f'Channel {channel+1}', fontsize=12)
        axs[0, channel].set_ylabel('Amplitude')
        axs[0, channel].set_xlabel(f'Time ({xlabel})')
        axs[0, channel].set_xlim(-0.05 * duration_scaled, 1.05 * duration_scaled)

        axs[1, channel].plot(freq_axis_signal, signal_amplitude, label='Amplitude spectrum', c='#FF4500', linewidth=0.4)
        axs[1, channel].vlines(freq_axis_signal, [0], signal_amplitude, color='#FF4500', linewidth=0.5) 
        axs[1, channel].scatter(freq_axis_signal, signal_amplitude, color='#FF4500', s=10) 
        axs[1, channel].set_ylabel('Magnitude')
        axs[1, channel].set_xlabel('Frequency (Hz)')
        # axs[1, channel].set_xlim(-10, 5000)

        # Formatting options (scientific notation)
        for row in range(2):
            axs[row, channel].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
            axs[row, channel].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axs[row, channel].get_yaxis().get_offset_text().set_x(-0.08) 
            axs[row, channel].grid(True)
            axs[row, channel].legend(loc='upper right')

    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.03, text_info, fontsize=10, ha='center', va='top', backgroundcolor='white')
    
    # Adjust the layout
    plt.subplots_adjust(left=0.1 if num_channels==1 else 0.085,\
                        right=0.9 if num_channels==1 else 0.98,\
                        top=0.93 if num_channels==1 else 0.9,\
                        bottom=0.11 if num_channels==1 else 0.12,\
                        hspace=0.17 if num_channels==1 else 0.25)
    plt.show()


def plot_hartley_spectrum(sample_rate, signal_data, transform_type='discrete'):
    if transform_type == 'discrete':
        signal_dht = apply_dht(signal_data, matrix_form=True)
    elif transform_type == 'fast':
        signal_dht = apply_fht(signal_data)

    duration_sec = get_duration_sec(sample_rate, signal_data)
    time_scale, xlabel = get_time_scale(duration_sec)
    t = get_time_axis(sample_rate, signal_data, time_scale)
    f = get_freq_axis(sample_rate*2, signal_dht)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    ax1.plot(t, signal_data, color='blue', linewidth=1)
    ax2.plot(f, signal_dht, color='#FF4500', linewidth=0.7)
    ax2.scatter(f, signal_dht, color='#FF4500', s=10) 
    ax2.vlines(f, [0], signal_dht, color='#FF4500', linewidth=0.5) 

    ax1.set_xlim(-0.1*t.max(), 1.1*t.max())
    ax1.set_ylim(1.5*signal_data.min(), 1.5*signal_data.max())
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel(f'Time ({xlabel})')
    ax1.grid(True)
    ax2.set_xlim(-0.1*sample_rate, 1.1*sample_rate)
    ax2.set_ylim(2*signal_dht.min(), 2*signal_dht.max())
    ax2.set_ylabel('Hartley Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def animate_hartley_spectrum(sample_rate, signal_data):
    signal_dht = apply_fht(signal_data)

    duration_sec = get_duration_sec(sample_rate, signal_data)
    time_scale, xlabel = get_time_scale(duration_sec)
    t = get_time_axis(sample_rate, signal_data, time_scale)
    f = get_freq_axis(2*sample_rate, signal_dht)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))

    line1, = ax1.plot(t, signal_data, color='blue', linewidth=1)
    line2, = ax2.plot(f, signal_dht, color='#FF4500', linewidth=0.7)
    scatter = ax2.scatter(f, signal_dht, color='#FF4500', s=10) 
    vlines = [Line2D([fx, fx], [0, fy], color='#FF4500', linewidth=0.5) for fx, fy in zip(f, signal_dht)]
    for vline in vlines:
        ax2.add_line(vline)

    ax1.set_xlim(-0.1*t.max(), 1.1*t.max())
    ax1.set_ylim(1.5*signal_data.min(), 1.5*signal_data.max())
    ax1.set_ylabel('Amplitude')
    ax1.set_xlabel(f'Time ({xlabel})')
    ax1.grid(True)
    ax2.set_xlim(-0.1*sample_rate, 1.1*sample_rate)
    ax2.set_ylim(2*signal_dht.min(), 2*signal_dht.max())
    ax2.set_ylabel('Hartley Spectrum')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.grid(True)

    def init():
        line1.set_data(t, signal_data)
        line2.set_data(f, signal_dht)
        scatter.set_offsets(np.column_stack([f, signal_dht]))
        for vline, (fx, fy) in zip(vlines, zip(f, signal_dht)):
            vline.set_xdata([fx, fx])
            vline.set_ydata([0, fy])
        return [line1, line2, scatter] + vlines


    def update(frame):
        shifted_signal = np.roll(signal_data, frame)
        shifted_dht = apply_fht(shifted_signal)

        line1.set_data(t, shifted_signal)
        line2.set_data(f, shifted_dht)
        scatter.set_offsets(np.column_stack([f, shifted_dht]))
        for vline, fy in zip(vlines, shifted_dht):
            vline.set_ydata([0, fy])

        return [line1, line2, scatter] + vlines
        
    ani = FuncAnimation(fig, update, frames=range(len(signal_data)), init_func=init, blit=True, interval=50)

    # Save the animation
    ani.save('hartley_transform_animation.gif', writer=PillowWriter(fps=20))

    plt.tight_layout()
    plt.show()
    # ani = FuncAnimation(fig, update, frames=range(len(signal_data)), init_func=init, blit=True, interval=50)

    # # Save the animation
    # writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('hartley_transform_animation.mp4', writer=writer)

    # plt.tight_layout()
    # plt.show()

    return ani

