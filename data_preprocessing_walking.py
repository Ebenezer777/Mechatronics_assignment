import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --------------------------
# 1. LOAD DATA
# --------------------------
a_df = pd.read_csv('./data/walking/Accelerometer.csv')
o_df = pd.read_csv('./data/walking/Orientation.csv')
g_df = pd.read_csv('./data/walking/Gyroscope.csv')

# --------------------------
# 2. STANDARDIZE COLUMN NAMES
# --------------------------
for df in [a_df, o_df, g_df]:
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)

# Accelerometer
acc_map = {"x": "acc_x", "y": "acc_y", "z": "acc_z"}
a_df.rename(columns=acc_map, inplace=True)

# Gyroscope
gyro_map = {"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"}
g_df.rename(columns=gyro_map, inplace=True)

# Orientation (Euler angles only)
orient_map = {"roll": "orient_x", "pitch": "orient_y", "yaw": "orient_z"}
o_df.rename(columns=orient_map, inplace=True)

# Remove quaternion & seconds_elapsed columns
o_df.drop(columns=[c for c in ["qx","qy","qz","qw"] if c in o_df.columns], inplace=True)
o_df.drop(columns=[c for c in o_df.columns if "seconds_elapsed" in c], inplace=True)

# --------------------------
# 3. MERGE SENSOR DATA
# --------------------------
data = pd.merge(a_df, o_df, on="time", how="inner")
data = pd.merge(data, g_df, on="time", how="inner")
print("Merged columns:", data.columns.tolist())

# --------------------------
# 4. REMOVE MISSING VALUES
# --------------------------
data = data.ffill().bfill()

# Ensure numeric
for col in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','orient_x','orient_y','orient_z']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.ffill().bfill()

# 5. OUTLIER REMOVAL (Z-score)

feature_cols = ['acc_x','acc_y','acc_z','orient_x','orient_y','orient_z','gyro_x','gyro_y','gyro_z']

def remove_outliers(df, cols, threshold=4):
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    mask = (np.abs(z) < threshold).all(axis=1)
    return df[mask]

# Copy raw data for comparison
raw_data = data.copy()

# Apply Z-score cleaning
data = remove_outliers(data, feature_cols)
data = data.reset_index(drop=True)

# --------------------------
# 6. BUILD RAW vs CLEANED TABLES
# --------------------------
def build_axis_table(raw_df, clean_df, col, name):
    return pd.DataFrame({
        "time": clean_df["time"],
        f"{name}_raw": raw_df[col].values[:len(clean_df)],
        f"{name}_clean": clean_df[col].values
    })

# Accelerometer
acc_x_table = build_axis_table(raw_data, data, "acc_x", "acc_x")
acc_y_table = build_axis_table(raw_data, data, "acc_y", "acc_y")
acc_z_table = build_axis_table(raw_data, data, "acc_z", "acc_z")

# Gyroscope
gyro_x_table = build_axis_table(raw_data, data, "gyro_x", "gyro_x")
gyro_y_table = build_axis_table(raw_data, data, "gyro_y", "gyro_y")
gyro_z_table = build_axis_table(raw_data, data, "gyro_z", "gyro_z")

# --------------------------
# 7. PLOT RAW vs CLEANED
# --------------------------
def plot_raw_vs_cleaned(table, axis_label, sensor_name):
    plt.figure(figsize=(12,4))
    plt.plot(table['time'], table[f"{axis_label}_raw"], label=f"{sensor_name} Raw", alpha=0.6)
    plt.plot(table['time'], table[f"{axis_label}_clean"], label=f"{sensor_name} Cleaned", alpha=0.8)
    plt.title(f"{sensor_name} {axis_label.upper()} Axis: Raw vs Cleaned")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{sensor_name} Value")
    plt.legend()
    plt.show()

for table, axis, sensor in [
    (acc_x_table,'acc_x','Accelerometer'),
    (acc_y_table,'acc_y','Accelerometer'),
    (acc_z_table,'acc_z','Accelerometer'),
    (gyro_x_table,'gyro_x','Gyroscope'),
    (gyro_y_table,'gyro_y','Gyroscope'),
    (gyro_z_table,'gyro_z','Gyroscope')
]:
    plot_raw_vs_cleaned(table, axis, sensor)

# --------------------------
# 8. LOW-PASS FILTER FUNCTION
# --------------------------
def low_pass_filter(df, col, fs=96, cutoff=4, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, df[col].values)
    return filtered

# Apply low-pass filter to accelerometer & gyroscope
fs = 96
cutoff = 4   # Hz, keep walking motion and remove high-frequency noise

for col in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']:
    data[col+'_filtered'] = low_pass_filter(data, col, fs, cutoff)

# --------------------------
# 9. PLOT RAW VS FILTERED (FULL DATA)
# --------------------------
def plot_raw_vs_filtered(df, col, sensor_name):
    plt.figure(figsize=(12,4))
    plt.plot(df['time'], df[col], label='Raw', alpha=0.6)
    plt.plot(df['time'], df[col+'_filtered'], label='Filtered', alpha=0.8)
    plt.title(f"{sensor_name} {col.upper()}: Raw vs Filtered (Full Data)")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{sensor_name} Value")
    plt.legend()
    plt.show()

for col, sensor in [
    ('acc_x','Accelerometer'), ('acc_y','Accelerometer'), ('acc_z','Accelerometer'),
    ('gyro_x','Gyroscope'), ('gyro_y','Gyroscope'), ('gyro_z','Gyroscope')
]:
    plot_raw_vs_filtered(data, col, sensor)

# --------------------------
# 10. PLOT RAW VS FILTERED FOR A SINGLE SEGMENT
# --------------------------
window_seconds = 2       # segment length in seconds
window_size = window_seconds * fs
start_idx = 500          # starting index (adjust as needed)
end_idx = start_idx + window_size

cols_to_visualize = ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']

for col in cols_to_visualize:
    plt.figure(figsize=(12,4))
    plt.plot(
        data['time'].iloc[start_idx:end_idx],
        data[col].iloc[start_idx:end_idx],
        label='Raw', alpha=0.6
    )
    plt.plot(
        data['time'].iloc[start_idx:end_idx],
        data[col+'_filtered'].iloc[start_idx:end_idx],
        label='Filtered', alpha=0.8
    )
    plt.title(f"{col.upper()}: Raw vs Filtered (Segmented {window_seconds}s window)")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{col} Value")
    plt.legend()
    plt.show()

from scipy.fft import fft, fftfreq
from scipy.signal import welch

# --------------------------
# 11. FREQUENCY-DOMAIN FEATURES
# --------------------------
def frequency_features(signal, fs=96):
    n = len(signal)
    
    # FFT
    yf = fft(signal)
    xf = fftfreq(n, 1/fs)
    
    # Take only positive frequencies
    xf = xf[:n//2]
    yf = np.abs(yf[:n//2])
    
    # Power Spectral Density (Welch)
    f_psd, Pxx = welch(signal, fs=fs, nperseg=256)
    
    # Spectral centroid
    spectral_centroid = np.sum(xf * yf) / np.sum(yf)
    
    # Spectral bandwidth (variance around centroid)
    spectral_bandwidth = np.sqrt(np.sum(((xf - spectral_centroid)**2) * yf) / np.sum(yf))
    
    # Spectral roll-off (frequency below which 85% of power is contained)
    cumulative_power = np.cumsum(yf**2)
    roll_off_threshold = 0.85 * cumulative_power[-1]
    spectral_rolloff = xf[np.where(cumulative_power >= roll_off_threshold)[0][0]]
    
    return {
        'fft_freqs': xf, 'fft_amplitude': yf,
        'psd_freqs': f_psd, 'psd_power': Pxx,
        'centroid': spectral_centroid,
        'bandwidth': spectral_bandwidth,
        'rolloff': spectral_rolloff
    }

# Example: compute for filtered accelerometer X (full segment)
fs = 96
segment_signal = data['acc_x_filtered'].iloc[start_idx:end_idx].values
features = frequency_features(segment_signal, fs)
# --------------------------
# 14. FFT & PSD FOR ALL FILTERED AXES
# --------------------------

axes = [
    'acc_x_filtered', 'acc_y_filtered', 'acc_z_filtered',
    'gyro_x_filtered', 'gyro_y_filtered', 'gyro_z_filtered'
]

for axis in axes:

    print(f"\nComputing FFT & PSD for: {axis}")

    # Take the same window as before (segment)
    signal = data[axis].iloc[start_idx:end_idx].values

    # Compute frequency features
    feats = frequency_features(signal, fs)

    # ---------- FFT PLOT ----------
    plt.figure(figsize=(10,4))
    plt.plot(feats['fft_freqs'], feats['fft_amplitude'])
    plt.title(f"FFT - {axis.upper()} (Filtered Segment)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.show()

    # ---------- PSD PLOT ----------
    plt.figure(figsize=(10,4))
    plt.semilogy(feats['psd_freqs'], feats['psd_power'])
    plt.title(f"PSD - {axis.upper()} (Filtered Segment)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.grid(True, alpha=0.3)
    plt.show()