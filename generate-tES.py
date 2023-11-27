import os, sys, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This script allows the user to generate tES synthetic data compatible with
the EEG framework developped by EEGdenoiseNet.

Currently supporting tDCS, tRNS, tACS. More methods coming soon...
"""

tES_mode = None
tES_supported_modes = ["tDCS", "tRNS", "tACS"]
V_COEFFS_lsm = [3.99, 3.64, 1.395, 9.92e-3, 3.35e-5]

def generate_tDCSvoltage_approximation(current, t, plot = False):
    """
    Generates tDCS voltage signal for a given current.
    """
    clean_signal = _generate_tDCSclean_voltage(current, t)
    noisy_signal = clean_signal + _generate_tDCSvoltage_noise(clean_signal)
    if plot: plot_tDCS_voltage_signal(clean_signal, noisy_signal, t)
    return noisy_signal
    
def _generate_tDCSclean_voltage(current, t):
    """
    Voltage signal generated using the proposed approximation in:
        ~ `Methods for extra-low voltage transcranial direct current stimulation: Current and time dependent impedance decreases` by Hahn et al.
    """
    return V_COEFFS_lsm[0] + V_COEFFS_lsm[1] * current + V_COEFFS_lsm[3] * current * t + 0.5 * V_COEFFS_lsm[4] * current * t**2

def _generate_tDCSvoltage_noise(clean_signal):
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

def generate_tRNSvoltage_approximation(peak_current, t, plot = False):
    """
    Generates tRNS voltage signal for a given peak current.
    
    The (random) current intensities are normally distributed 
    with 99% of the values lying between the peak-to-peak amplitude :)
    """
    current = np.random.normal(0, peak_current/3, size = t.shape[0])
    voltage = _generate_tRNS_voltage(current, t)
    if plot: plot_tRNS_signals(current, voltage, t)
    return voltage

def _generate_tRNS_voltage(current, t):
    """
    Voltage signal generated using the proposed approximation in:
        ~ `Methods for extra-low voltage transcranial direct current stimulation: Current and time dependent impedance decreases` by Hahn et al.
    """
    return V_COEFFS_lsm[0] + V_COEFFS_lsm[1] * current + V_COEFFS_lsm[3] * current * t + 0.5 * V_COEFFS_lsm[4] * current * t**2

def plot_tRNS_signals(current, voltage, t):
    """
    Utility method to plot the tRNS signals generated.
    """
    f, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax[0].plot(t, current, color = 'blue'), ax[0].title.set_text('tRNS Current (in mA)') ,\
    ax[0].set_xlabel('Time (s)'), ax[0].set_ylabel('Current (mA)'), ax[0].grid(True)
    ax[1].plot(t, voltage, color = 'red'), ax[1].title.set_text('tRNS Voltage (in V)') ,\
    ax[1].set_xlabel('Time (s)'), ax[1].set_ylabel('Voltage (V)'), ax[1].grid(True)
    plt.show()

def generate_tACSvoltage_approximation(amplitude, frequency, t, plot = False):
    """
    Generates tACS voltage signal for a given amplitude current and frequency.
    """
    current = amplitude * np.sin(2 * np.pi * frequency * t)
    voltage_clean = _generate_tACS_voltage(current, t)
    voltage_noisy = voltage_clean + _generate_tACSvoltage_noise(voltage_clean)
    if plot: plot_tACS_signals(current, voltage_clean, voltage_noisy, t)
    return voltage_noisy

def _generate_tACS_voltage(current, t):
    """
    Voltage signal generated using the proposed approximation in:
        ~ `Methods for extra-low voltage transcranial direct current stimulation: Current and time dependent impedance decreases` by Hahn et al.
    """
    return V_COEFFS_lsm[0] + V_COEFFS_lsm[1] * current + V_COEFFS_lsm[3] * current * t + 0.5 * V_COEFFS_lsm[4] * current * t**2

def _generate_tACSvoltage_noise(clean_signal):
    """
    Random gaussian noise is added to the generated clean voltages. For increased
    diversity in the tACS signals, the std deviation of the noise is selected randomly
    between a set of values.
    """
    noise_std = random.choice([0.1, 0.5, 1])
    return np.random.normal(0, noise_std, clean_signal.shape[0])

def plot_tACS_signals(current, voltage_clean, voltage_noisy, t):
    """
    Utility method to plot the tACS signals generated.
    """
    f, ax = plt.subplots(1, 2, figsize=(10, 7))
    ax[0].plot(t, current, color = 'blue'), ax[0].title.set_text('tACS Current (in mA)') ,\
    ax[0].set_xlabel('Time (s)'), ax[0].set_ylabel('Current (mA)'), ax[0].grid(True)

    ax[1].plot(t, voltage_clean, label = 'Clean Signal', color = 'blue'), ax[1].plot(t, voltage_noisy, label = 'Noisy Signal', color = 'red'),\
    ax[1].title.set_text('tACS Voltage (in V)') , ax[1].set_xlabel('Time (s)'), ax[1].set_ylabel('Voltage (V)'), ax[1].grid(True), ax[1].legend(loc = 'upper center')
    plt.show()

if __name__ == "__main__":
    print("Welcome to tES data generation...")
    try:
        tES_mode = sys.argv[1] # Mode of tES operation
        if tES_mode not in tES_supported_modes:
            raise ValueError(f"Selected tES mode not supported, please try one of the following: {tES_supported_modes}") 
        # Generate sample times
        start_time = 0
        end_time = int(input("Please enter the desired end time (in s): "))
        sampling_rate = int(input("Please enter the sampling rate: "))
        t = np.linspace(start_time, end_time, end_time * sampling_rate, endpoint = False)
        if tES_mode == "tDCS":
            current = float(input("Please enter the desired constant current in mA: "))
            num_samples = int(input("Please enter the desired number of samples: "))
            # Get voltage approximations
            tDCS_samples = np.empty((num_samples, t.size))
            for sample in tqdm(range(num_samples)):
                tDCS_samples[sample, :] = generate_tDCSvoltage_approximation(current, t)
            # Store the results in EEGdenoiseNet compatible format
            np.save(f"tDCS_all_epochs-{current}mA.npy", tDCS_samples * 1e6)
        elif tES_mode == "tRNS":
            peak_current = float(input("Please enter the desired peak current in mA: "))
            num_samples = int(input("Please enter the desired number of samples: ")) 
            # Get voltage approximations
            tRNS_samples = np.empty((num_samples, t.size))
            for sample in tqdm(range(num_samples)):
                tRNS_samples[sample, :] = generate_tRNSvoltage_approximation(peak_current, t)
            # Store the results in EEGdenoiseNet compatible format
            np.save(f"tRNS_all_epochs-{peak_current}mA.npy", tRNS_samples * 1e6)
        elif tES_mode == "tACS":
            amplitude_current = float(input("Please enter the desired amplitude current in mA: "))
            frequency = float(input("Please enter the desired frequency in Hz: ")) 
            num_samples = int(input("Please enter the desired number of samples: "))
            # Get voltage approximations
            tACS_samples = np.empty((num_samples, t.size))
            for sample in tqdm(range(num_samples)):
                tACS_samples[sample, :] = generate_tACSvoltage_approximation(amplitude_current, frequency, t)
            # Store the results in EEGdenoiseNet compatible format
            np.save(f"tACS_all_epochs-{amplitude_current}mA_{frequency}Hz.npy", tACS_samples * 1e6)  
        print("Finished generating data samples!")
    except IndexError:
        raise ValueError(f"Please, provide a tES mode. Currently supporting {tES_supported_modes}")