import numpy as np
import scipy.signal
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

"""
Quality assessment metrics for EEG denoising methods
"""

def covariance(x,y):
    """ 
    Returns the covariance between x and y.
    """
    return np.cov(x,y)[0,1]

def variance(x):
    """
    Returns the variance of x.

    Note: Using the implementation in the built-in statistics module fails
    due to a well-documented casting issue between 32-bit and 64-bit floats.
    (See: https://bugs.python.org/issue39218)
    """
    return np.var(x)

def rms(x):
    """ 
    Returns the RMS (Root Mean Squared) value of x as defined in eq. 3 of: 
        ~'Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., & Liu, Q. (2021). 
        EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG 
        denoising. Journal of Neural Engineering, 18(5), 056057.'~
    """
    return np.sqrt(np.divide(np.sum(np.power(x,2)), len(x)))

def rrmse_temporal(denoised, ground_truth):
    """ 
    Returns the RRMSE (Relative Root Mean Squared Error), in the *temporal* domain, 
    between the denoised values [f(y)] and the clean groud truths [x] as defined in 
    eq. 7 of:
        ~'Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., & Liu, Q. (2021). 
        EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG 
        denoising. Journal of Neural Engineering, 18(5), 056057.'~
    """
    return np.divide(rms(denoised - ground_truth), rms(ground_truth))

def rrmse_spectral(denoised, ground_truth, sampling_freq = 256):
    """ 
    Returns the RRMSE (Relative Root Mean Squared Error), in the *spectral* domain, 
    between the denoised values [f(y)] and the clean groud truths [x] as defined in 
    eq. 8 of:
        ~'Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., & Liu, Q. (2021). 
        EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG 
        denoising. Journal of Neural Engineering, 18(5), 056057.'~
    """
    frequencies_denoised, PSD_denoised = scipy.signal.periodogram(x = denoised, fs = sampling_freq, scaling = 'density')
    frequencies_ground_truth, PSD_ground_truth = scipy.signal.periodogram(x = ground_truth, fs = sampling_freq, scaling = 'density')
    return np.divide(rms(PSD_denoised - PSD_ground_truth), rms(PSD_ground_truth))

def cc(denoised, ground_truth):
    """ 
    Returns the CC (Correlation Coefficient) between the denoised values [f(y)] 
    and the clean groud truths [x] as defined in eq. 9 of:
        ~'Zhang, H., Zhao, M., Wei, C., Mantini, D., Li, Z., & Liu, Q. (2021). 
        EEGdenoiseNet: a benchmark dataset for deep learning solutions of EEG 
        denoising. Journal of Neural Engineering, 18(5), 056057.'~
    """
    return np.divide(covariance(denoised, ground_truth), np.sqrt(variance(denoised)*variance(ground_truth)))