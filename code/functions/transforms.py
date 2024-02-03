import numpy as np
import time


def apply_fft(signal_data):
    # Extract amplitude and phase spectra
    
    start = time.time()
    signal_fft = np.fft.fft(signal_data, axis=0)
    end = time.time()
    
    amplitude = np.abs(signal_fft)
    phase = np.angle(signal_fft)
    
    print(f"Running time `numpy`s FFT: \t {end - start} sec")

    return amplitude, phase


def inverse_fft(amplitude, phase):
    # Reconstruct signal from amplitude and phase spectra
    signal_fft = amplitude * np.exp(1j * phase)
    
    start = time.time()
    signal_data = np.fft.ifft(signal_fft, axis=0)
    end = time.time()
    
    # print(f"Running time `numpy`s inverse FFT: \t {end - start:.8f} sec")

    return signal_data.real


def apply_dft(signal_data):
    # Implement discrete Fourier transform 
    start = time.time()
    signal_dft = np.zeros(signal_data.shape, dtype=complex)
    n = signal_data.shape[0]

    for u in np.arange(n):
        for x in range(n):
            signal_dft[u] += signal_data[x] * np.exp(-1j * 2 * np.pi * (u * x) / n)
    end = time.time()

    amplitude = np.abs(signal_dft)
    phase = np.angle(signal_dft)

    print(f"Running time DFT: \t {np.round(end - start, 8):.8f} sek.")

    return amplitude, phase    


def inverse_dft(amplitude, phase):
    # Implement inverse discrete Fourier transform
    signal_dft = amplitude * np.exp(1j * phase)
    signal_data = np.zeros(signal_dft.shape, dtype=complex)
    n = signal_data.shape[0]

    start = time.time()
    for x in np.arange(n):
        for u in range(n):
            signal_data[x] += signal_dft[u] * np.exp(1j * 2 * np.pi * (u * x) / n)
    end = time.time()
    
    # print(f"Running time inverse DFT: \t {end - start:.8f} sec")

    return signal_data.real / n


def cas(x):
    return np.cos(x) + np.sin(x)


def apply_dht(signal_data, matrix_form=True):
    # Implement discrete Hartley transform 
    start = time.time()
    signal_dht = np.zeros(signal_data.shape, dtype=float)
    N = signal_data.shape[0]

    if matrix_form == True:
        u = np.arange(N)
        x = u.reshape((N, 1))
        hartley_matrix = 1/np.sqrt(N) * cas(2 * np.pi * u * x / N)
        signal_dht = np.dot(hartley_matrix, signal_data)
    
    else:
        for u in np.arange(N):
            for x in range(N):
                signal_dht[u] += signal_data[x] * cas(2 * np.pi * (u * x) / N)

    end = time.time()
    print(f"Running time DHT: \t {np.round(end - start, 8):.8f} sek.")

    return signal_dht


def inverse_dht(signal_dht, matrix_form=True):
    # Implement inverse discrete Hartley transform
    start = time.time()
    signal_data = np.zeros(signal_dht.shape, dtype=float)
    n = signal_data.shape[0]

    if matrix_form == True:
        signal_data = apply_dht(signal_dht, matrix_form=True)

    else:
        for x in np.arange(n):
            for u in range(n):
                signal_data[x] += signal_dht[u] * cas(2 * np.pi * (u * x) / n)
    
    end = time.time()
    # print(f"Running time inverse DHT: \t {end - start:.8f} sec")

    return signal_data


def apply_fht(signal_data):
    # Implement fast Fourier transform 
    start = time.time()
    signal_fft = np.fft.fft(signal_data)
    signal_fht = np.real(signal_fft) - np.imag(signal_fft)
    end = time.time()
    print(f"Running time FHT: \t {end - start:.8f} sec")
    return signal_fht


def inverse_fht(signal_fht):
    # Implement inverse fast Fourier transform 
    start = time.time()
    n = signal_fht.shape[0]
    signal_data = apply_fht(signal_fht)
    signal_data = 1.0/n*signal_data
    end = time.time()
    # print(f"Running time inferse FHT: \t {end - start:.8f} sec")
    return signal_data