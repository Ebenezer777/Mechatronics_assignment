import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch


# ----------------------------------------------------------
# 1. Frequency Feature Extraction
# ----------------------------------------------------------
def compute_frequency_features(signal, fs=96):

    # ---------- FFT ----------
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1 / fs)

    # Keep positive frequencies only
    xf = xf[:n // 2]
    yf = np.abs(yf[:n // 2])

    # ---------- PSD (Welch) ----------
    f_psd, Pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    # ---------- Spectral Centroid ----------
    spectral_centroid = np.sum(xf * yf) / np.sum(yf)

    # ---------- Spectral Bandwidth ----------
    spectral_bandwidth = np.sqrt(np.sum(((xf - spectral_centroid) ** 2) * yf) / np.sum(yf))

    # ---------- Spectral Roll-off (85%) ----------
    cumulative_power = np.cumsum(yf ** 2)
    rolloff_threshold = 0.85 * cumulative_power[-1]
    idx = np.where(cumulative_power >= rolloff_threshold)[0][0]
    spectral_rolloff = xf[idx]

    return {
        'fft_freqs': xf,
        'fft_amplitude': yf,
        'psd_freqs': f_psd,
        'psd_power': Pxx,
        'centroid': spectral_centroid,
        'bandwidth': spectral_bandwidth,
        'rolloff': spectral_rolloff
    }


# ----------------------------------------------------------
# 2. Plot FFT
# ----------------------------------------------------------
def plot_fft(features, title="FFT Amplitude Spectrum"):
    plt.figure(figsize=(10, 4))
    plt.plot(features['fft_freqs'], features['fft_amplitude'])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.show()


# ----------------------------------------------------------
# 3. Plot PSD
# ----------------------------------------------------------
def plot_psd(features, title="Power Spectral Density (PSD)"):
    plt.figure(figsize=(10, 4))
    plt.semilogy(features['psd_freqs'], features['psd_power'])
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.show()


# ----------------------------------------------------------
# 4. Compare Two FFTs (walking vs stair climbing)
# ----------------------------------------------------------
def compare_fft(f1, f2, label1="Walking", label2="Stair Climbing"):
    plt.figure(figsize=(10, 4))
    plt.plot(f1['fft_freqs'], f1['fft_amplitude'], label=label1)
    plt.plot(f2['fft_freqs'], f2['fft_amplitude'], label=label2)
    plt.title("FFT Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ----------------------------------------------------------
# 5. Compare Two PSDs (walking vs stair climbing)
# ----------------------------------------------------------
def compare_psd(f1, f2, label1="Walking", label2="Stair Climbing"):
    plt.figure(figsize=(10, 4))
    plt.semilogy(f1['psd_freqs'], f1['psd_power'], label=label1)
    plt.semilogy(f2['psd_freqs'], f2['psd_power'], label=label2)
    plt.title("PSD Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()