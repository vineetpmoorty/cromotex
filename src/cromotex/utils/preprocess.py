import wfdb
import wfdb.processing as wp
import numpy as np
from scipy.signal import resample_poly
import scipy


def ecg_consistency(signal: np.ndarray, fields, target_lead_order = None):
    # signal: numpy array of shape (n_samples, n_leads)
    # current_fields: list of strings, the current order of the leads

    if not np.all(np.array(fields['units']) == 'mV'):
        raise ValueError("ECG units not in mV")

    if target_lead_order is None:
        target_lead_order = [
            'I', 'II', 'III', 'aVR', 'aVF', 'aVL',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]

    current_lead_order = fields['sig_name']

    if len(current_lead_order) != len(target_lead_order):
        raise ValueError("Number of ECG leads does not meet target")

    if np.all(np.array(current_lead_order) == np.array(target_lead_order)):
        return signal
    else:
        # Create a mapping from current fields to target fields
        field_map = {
            field: idx for idx, field in enumerate(current_lead_order)
        }
        target_order = [field_map[field] for field in target_lead_order]
        return signal[:, target_order]
    return signal

def normalize_per_lead(ecg_array, min_val=-1.0, max_val=1.0):
    """
    Perform per-lead min-max normalization on an ECG array.
    Parameters:
        ecg_array (np.ndarray): Input ECG data of shape [n_samples, n_leads].
    """
    
    lead_min = np.min(ecg_array, axis=0, keepdims=True)  # Shape [1, n_leads]
    lead_max = np.max(ecg_array, axis=0, keepdims=True)  # Shape [1, n_leads]

    range_lead = lead_max - lead_min
    range_lead[range_lead == 0] = 1e-8 

    normalized_ecg = (ecg_array - lead_min) / range_lead  # Normalize to [0, 1]
    # Scale to [min_val, max_val]
    normalized_ecg = normalized_ecg * (max_val - min_val) + min_val 
    return normalized_ecg

def resample_signal_poly(signal, original_fs, target_fs=100):
    """
    Resample a signal to a new sampling frequency using polyphase filtering.

    Parameters:
        signal (numpy.ndarray): Input signal of shape [n_samples, n_channels].
        original_fs (float): Original sampling frequency in Hz.
        target_fs (float): Target sampling frequency in Hz (default: 100 Hz).
    """
    # Compute the upsampling and downsampling factors
    up = target_fs
    down = original_fs
    
    # Resample each channel
    resampled_signal = np.zeros((int(signal.shape[0] * up / down), signal.shape[1]))
    for i in range(signal.shape[1]):
        resampled_signal[:, i] = resample_poly(signal[:, i], up, down)
    return resampled_signal

def baseline_wander_removal(data, sampling_frequency = 100):
    """
    https://github.com/jntorres/ecg_preprocess/blob/main/preprocess_ecgs_00.py
    """ 
    row, __ = data.shape
    processed_data = np.zeros(data.shape)
    for lead in range(0, row):
        # Baseline estimation
        win_size = int(np.round(0.2 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(data[lead, :], win_size)
        win_size = int(np.round(0.6 * sampling_frequency)) + 1
        baseline = scipy.signal.medfilt(baseline, win_size)
        # Removing baseline
        filt_data = data[lead, :] - baseline
        processed_data[lead, :] = filt_data
    return processed_data
