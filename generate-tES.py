import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This script allows the user to generate tES synthetic data compatible with
the EEG framework developped by EEGdenoiseNet.

Currently supporting tDCS. More methods coming soon...
"""

tES_mode = None
tES_supported_modes = ["tDCS"]
V_COEFFS_lsm_tDCS = [3.99, 3.64, 1.395, 9.92e-3, 3.35e-5]

def generate_voltage_approximation(current, t, plot = False):
    """
    Generates tDCS voltage signal for a given current.
    """
    clean_signal = _generate_clean_voltage(current, t)
    noisy_signal = clean_signal + _generate_voltage_noise(clean_signal)
    if plot: plot_tDCS_voltage_signal(clean_signal, noisy_signal, t)
    return noisy_signal
    
def _generate_clean_voltage(current, t):
    """
    Voltage signal generated using the proposed approximation in:
        ~ `Methods for extra-low voltage transcranial direct current stimulation: Current and time dependent impedance decreases` by Hahn et al.
    """
    return V_COEFFS_lsm_tDCS[0] + V_COEFFS_lsm_tDCS[1] * current + V_COEFFS_lsm_tDCS[3] * current * t + 0.5 * V_COEFFS_lsm_tDCS[4] * current * t**2

def _generate_voltage_noise(clean_signal):
    """
    Random gaussian noise is added to the generated clean voltages. For increased
    diversity in the tDCS signals, the std deviation of the noise is selected randomly
    between 0.01, 0.001, and 0.0001.
    """
    noise_std = random.choice([0.01, 0.001, 0.0001])
    return np.random.normal(0, noise_std, clean_signal.shape[0])

def plot_tDCS_voltage_signal(signal_clean, signal_noise, t, title = None):
    """
    Utility method to plot the clean/noisy signals generated.
    """
    plt.figure(figsize = (10, 7)) 
    plt.plot(t, signal_clean, label = 'Clean Signal', color='blue')
    plt.plot(t, signal_noise, label = 'Noisy Signal', color='red')
    plt.title(title if title else 'tDCS Signal (in V)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    print("Welcome to tES data generation...")
    try:
        tES_mode = sys.argv[1] # Mode of tES operation
        if tES_mode not in tES_supported_modes:
            raise ValueError(f"Selected tES mode not supported, please try one of the following: {tES_supported_modes}") 
        if tES_mode == "tDCS":
            current = float(input("Please enter the desired constant current in mA: "))
            num_samples = int(input("Please enter the desired number of samples: "))
            # Generate sample times
            start_time, end_time, sampling_rate = 0, 2, 256
            t = np.linspace(start_time, end_time, end_time * sampling_rate, endpoint = False)
            # Get voltage approximations
            tDCS_samples = np.empty((num_samples, t.size))
            for sample in tqdm(range(num_samples)):
                tDCS_samples[sample, :] = generate_voltage_approximation(current, t)
        np.save(f"tDCS_all_epochs-{current}mA.npy", tDCS_samples) # Store the results in EEGdenoiseNet compatible format
        print("Finished generating data samples!")
    except IndexError:
        raise ValueError(f"Please, provide a tES mode. Currently supporting {tES_supported_modes}")