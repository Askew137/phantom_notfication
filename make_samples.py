import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter
import os

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_pink_noise(duration_s, fs=44100, volume_db=-25):
    """Generates pink noise using FFT."""
    samples = int(duration_s * fs)
    white_noise = np.random.randn(samples)
    
    # FFT method for pink noise (1/f)
    X = np.fft.rfft(white_noise)
    f = np.fft.rfftfreq(samples)
    f[0] = f[1]  # Avoid division by zero at DC
    X = X / np.sqrt(f)
    pink_filtered = np.fft.irfft(X, n=samples)
    
    # Normalize and apply volume
    pink_filtered /= np.max(np.abs(pink_filtered))
    amplitude = 10**(volume_db / 20)
    return pink_filtered * amplitude

def load_wav(filename):
    """Loads a wav file and normalizes it to float32."""
    fs, data = wavfile.read(filename)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    # Convert stereo to mono if necessary
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return fs, data

# Configuration
FS = 44100
DURATION = 6  # seconds
NOISE_VOL = -25 # dB
VIBE_LEVELS = [-55, -52, -49, -46, -43] # dB relative to original
INSERT_TIMES = [2, 3, 4] # seconds

# 1. Load the vibration
vibration_path = os.path.join(SCRIPT_DIR, "vibration.wav")
fs_vibe, vibe_data = load_wav(vibration_path)

# 2. Generate base noise
noise = generate_pink_noise(DURATION, fs=FS, volume_db=NOISE_VOL)

# 3. Export Level 0 (Noise only)
out_path_0 = os.path.join(SCRIPT_DIR, "stimulus_lvl_0.wav")
wavfile.write(out_path_0, FS, (noise * 32767).astype(np.int16))
print("Generated Level 0")

# 4. Generate Levels 1-5
for i, gain_db in enumerate(VIBE_LEVELS, 1):
    # Scale vibration
    gain_linear = 10**(gain_db / 20)
    scaled_vibe = vibe_data * gain_linear
    
    for insert_time in INSERT_TIMES:
        # Create a copy of the noise and overlay the vibration at insert_time
        output = noise.copy()
        start_idx = int(insert_time * FS)
        end_idx = start_idx + len(scaled_vibe)
        
        # Ensure we don't exceed array bounds
        if end_idx <= len(output):
            output[start_idx:end_idx] += scaled_vibe
        else:
            fit_len = len(output) - start_idx
            if fit_len > 0:
                output[start_idx:len(output)] += scaled_vibe[:fit_len]
        
        # Normalize final result to prevent clipping before export
        if np.max(np.abs(output)) > 1.0:
            output /= np.max(np.abs(output))
            
        filename = f"stimulus_lvl_{i}_s_{insert_time}.wav"
        out_path = os.path.join(SCRIPT_DIR, filename)
        wavfile.write(out_path, FS, (output * 32767).astype(np.int16))
        print(f"Generated {filename} (gain {gain_db} dB, time {insert_time}s)")
