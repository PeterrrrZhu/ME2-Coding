import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#Discrete Fourier Transform (DFT)
def DFT(yn):
    """
    Calculates the Discrete Fourier Transform manually.
    yn: A 1D array of numbers representing the signal in the time domain.
    """
    # Find out how many data points (N) we have in our time signal
    N = len(yn) 
    
    # Calculate the fundamental angular step size (omega)
    w = 2 * np.pi / N 
    
    # Create an array of complex numbers (0 + 0j) to store our final results.
    # We use 'complex' because Fourier transforms deal with magnitude and phase.
    FTk = np.zeros(N, dtype=complex)
    
    # Loop through each frequency bin 'k'
    for k in range(0, N):
        # For every frequency, loop through each point in time 'n'
        for n in range(0, N):
            # Apply the DFT mathematical formula: sum( y[n] * e^(-i*k*w*n) )
            # In Python, '1j' represents the imaginary unit 'i'
            FTk[k] += np.exp(-1j * k * w * n) * yn[n]
            
    # Return the frequency domain data
    return FTk


# =====================================================================
# YOUR MAIN TRANSFORM SCRIPT
# =====================================================================
def Transform(u_history, n_r, dt, output_dir="outputs"):
    """
    Extracts the vibration signal at a specific point on the membrane 
    and converts it from the time domain to the frequency domain using the DFT.
    """
    
    # --- 1. Select an observation point ---
    # We pick a point halfway along the radius.
    # The '//' symbol means integer division (it divides and rounds down to a whole number).
    # We do this because array indices must be whole numbers.
    r_idx = n_r // 2 
    
    # We just pick the first angle (index 0)
    theta_idx = 0 

    # Extract the displacement over time at this specific (r, theta) location.
    # The ':' means "give me all the time steps" for this specific r and theta.
    # This gives us a 1D list of numbers representing how that point moved up and down.
    time_signal = u_history[r_idx, theta_idx, :]

    # --- 2. Perform Temporal Discrete Fourier Transform (DFT) ---
    # Find out the total number of time samples
    N = len(time_signal) 

    # Use the teacher's custom DFT function instead of numpy's built-in FFT
    u_fft = DFT(time_signal)

    # The DFT gives us complex numbers, but we just want the physical amplitude of the vibrations.
    # We use np.abs() to calculate the magnitude (absolute value) of the complex numbers.
    # We only take the first half of the array ([:N // 2]) because the second half is just a mirrored copy (Nyquist symmetry).
    u_amplitude = np.abs(u_fft[:N // 2]) 

    # --- 3. Frequency Mapping ---
    # fs is the sampling frequency: how many samples (frames) we recorded per second.
    fs = 1.0 / dt 

    # Create the x-axis (frequencies in Hz).
    # np.linspace creates an array of evenly spaced numbers from 0 up to half the sampling frequency.
    freq_axis = np.linspace(0, fs / 2, N // 2)

    # --- 4. Visualization (Plotting) ---
    # Create a new blank canvas for the plot, 10 inches wide and 5 inches tall
    plt.figure(figsize=(10, 5))
    
    # Plot the frequencies on the x-axis and their amplitudes on the y-axis
    plt.plot(freq_axis, u_amplitude)
    
    # Add titles and labels so the chart is easy to read
    plt.title(f'Frequency Spectrum at r_idx={r_idx}, theta_idx={theta_idx}')
    
    # Limit the x-axis to show only 0 to 1000 Hz so we can zoom in on the important peaks
    plt.xlim(0, 1000) 
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    
    # Turn on the background grid lines to make it easier to read the values
    plt.grid(True) 
    
    # --- 5. Saving the File ---
    # Set up the folder path where the image will be saved
    output_path = Path(output_dir) / "fourier_spectrum_dft.png"
    
    # Make sure the 'outputs' folder actually exists. If it doesn't, create it.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the picture to the folder with a resolution of 150 dpi (dots per inch)
    plt.savefig(output_path, dpi=150)
    print(f"Fourier plot successfully saved to {output_path}")
    
    # Close the plot window internally to free up the computer's memory
    plt.close()