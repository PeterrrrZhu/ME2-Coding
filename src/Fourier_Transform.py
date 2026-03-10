import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def Transform(u_history, n_r, dt, output_dir="outputs"):
    #u_history = Result["u_history"]
    #n_r = u_history.shape[0]
    #dt = default_parameters()["dt"]

    # --- 1. Setup Parameters (Use the same values from your PDE solver) ---
    # dt = ... (Your time step)
    # n_r, n_theta, n_steps_plus_1 = u_history.shape

    # --- 2. Select an observation point ---
    # We pick a point at middle radius to capture the vibration clearly
    # Avoiding r=0 might be safer if there's numerical noise at the singularity
    r_idx = n_r // 2 
    theta_idx = 0 

    # Extract the displacement over time at this specific (r, theta) location
    # shape: (n_steps + 1,)
    time_signal = u_history[r_idx, theta_idx, :]

    # --- 3. Perform Temporal Fast Fourier Transform (FFT) ---
    # N is the total number of time samples
    N = len(time_signal)

    # Compute the FFT (returns complex numbers)
    u_fft = np.fft.fft(time_signal)

    # Compute the magnitude (amplitude) of the frequencies
    # We use only the first half of the array due to Nyquist symmetry
    u_amplitude = np.abs(u_fft[:N // 2]) 

    # --- 4. Frequency Mapping ---
    # fs is the sampling frequency: how many samples per second
    fs = 1.0 / dt 

    # Generate the frequency axis (Hz)
    # Mapping: f_k = k * fs / N
    freq_axis = np.linspace(0, fs / 2, N // 2)

    # --- 5. Visualization ---
    plt.figure(figsize=(10, 5))
    plt.plot(freq_axis, u_amplitude)
    plt.title(f'Frequency Spectrum at r_idx={r_idx}, theta_idx={theta_idx}')
    plt.xlim(0,1000)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    output_path = Path(output_dir) / "fourier_spectrum.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Fourier plot successfully saved to {output_path}")
    
    plt.close() # Release memory by closing the plot after saving

    # --- Optional: 2D Spatial FFT at a fixed time step 'n' ---
    # If you want to see the spatial frequency (wave numbers) at the final state:
    # u_spatial_fft = np.fft.fft2(u_history[:, :, -1])
    # u_spatial_amp = np.abs(np.fft.fftshift(u_spatial_fft))
