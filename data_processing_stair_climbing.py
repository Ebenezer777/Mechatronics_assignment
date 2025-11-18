import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.signal import welch

# --------------------------
# 1. LOAD DATA (STAIR CLIMBING)
# --------------------------
a_df = pd.read_csv('./data/stair_climbing/Accelerometer.csv')
o_df = pd.read_csv('./data/stair_climbing/Orientation.csv')
g_df = pd.read_csv('./data/stair_climbing/Gyroscope.csv')

# --------------------------
# 2. STANDARDIZE COLUMN NAMES
# --------------------------
for df in [a_df, o_df, g_df]:
    if "timestamp" in df.columns:
        df.rename(columns={"timestamp": "time"}, inplace=True)

# Accelerometer
a_df.rename(columns={"x":"acc_x","y":"acc_y","z":"acc_z"}, inplace=True)
# Gyroscope
g_df.rename(columns={"x":"gyro_x","y":"gyro_y","z":"gyro_z"}, inplace=True)
# Orientation (Euler angles only)
o_df.rename(columns={"roll":"orient_x","pitch":"orient_y","yaw":"orient_z"}, inplace=True)
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
for col in ['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z','orient_x','orient_y','orient_z']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.ffill().bfill()

# --------------------------
# 5. OUTLIER REMOVAL (Z-score)
# --------------------------
feature_cols = ['acc_x','acc_y','acc_z','orient_x','orient_y','orient_z','gyro_x','gyro_y','gyro_z']
def remove_outliers(df, cols, threshold=4):
    z = (df[cols] - df[cols].mean()) / df[cols].std()
    mask = (np.abs(z) < threshold).all(axis=1)
    return df[mask]

raw_data = data.copy()
data = remove_outliers(data, feature_cols)
data = data.reset_index(drop=True)

# --------------------------
# 6. LOW-PASS FILTER FUNCTION
# --------------------------
def low_pass_filter(df, col, fs=96, cutoff=4, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered = filtfilt(b, a, df[col].values)
    return filtered

fs = 96
cutoff = 4
for col in ['acc_x','acc_y','acc_z','orient_x','orient_y','orient_z']:
    data[col+'_filtered'] = low_pass_filter(data, col, fs, cutoff)

