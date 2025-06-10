"""
Sources:
    1. https://github.com/svthapa/MoRE/blob/main/utils/ecg_augmentations.py
    2. https://github.com/Jwoo5/fairseq-signals/blob/master/
    fairseq_signals/data/ecg/perturb_ecg_dataset.py
"""
import numpy as np 
import random
import numpy as np
import random
import math
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

def time_warp(segment, target_length):
    x = np.linspace(0, 1, len(segment))
    f = interp1d(x, segment, kind='linear')
    x_new = np.linspace(0, 1, target_length)
    return f(x_new)

def process_ecg_channel(channel, m, w):
    segment_length = len(channel) // m
    modified_channel = []
    segments_to_modify = random.sample(range(m), m // 2)

    for i in range(m):
        segment = channel[i * segment_length: (i+1) * segment_length]
        target_length = segment_length + int(
            segment_length * w
            if i in segments_to_modify
            else -segment_length * w
        )
        
        # Adjust the last segment to ensure the total length is 5000
        if i == m - 1:
            target_length = 1000 - sum(len(s) for s in modified_channel)

        modified_segment = time_warp(segment, target_length)
        modified_channel.append(modified_segment)

    return np.concatenate(modified_channel)

def time_warp_ecg(ecg_data, m=4, w=0.25):
    if m % 2 != 0:
        raise ValueError("m must be an even number.")

    modified_ecg_data = np.array(
        [process_ecg_channel(channel, m, w) for channel in ecg_data]
    )
    return modified_ecg_data


def permutation_augmentation(ecg_signal, m=4):
    """
    Apply permutation augmentation on the given
    ECG signal, preserving channel order.

    Parameters:
    - ecg_signal: The input ECG signal with shape (num_channels, num_samples)
    - m: The number of segments to divide each channel into

    Returns:
    - Augmented signal with shape (num_channels, num_samples)
    """
    # Get the number of channels and samples

    num_channels, num_samples = ecg_signal.shape
    
    # Check if the signal length is divisible by 'm'
    if num_samples % m != 0:
        raise ValueError("Signal length is not divisible by 'm'")
    
    # Calculate the length of each segment within a channel
    segment_length = num_samples // m
    
    # Initialize an empty array for the augmented signal
    augmented_signal = np.empty_like(ecg_signal)
    
    # Divide each channel into 'm' segments, shuffle them, and concatenate back
    for channel in range(num_channels):
        channel_data = ecg_signal[channel, :]
        segments = [
            channel_data[i * segment_length : (i + 1) * segment_length]
            for i in range(m)
        ]
        np.random.shuffle(segments)
        augmented_signal[channel, :] = np.concatenate(segments)
    
    return augmented_signal


def add_gaussian_noise(ecg_data, sigma):
    """
    Adds Gaussian noise to the ECG data.

    Parameters:
    - ecg_data: The ECG signal data.
    - sigma: The standard deviation of the Gaussian noise.

    Returns:
    - augmented: The ECG signal with added Gaussian noise.
    """
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    noise = np.random.normal(loc=0, scale=sigma, size=ecg_data.shape)
    
    # Add the noise to the ECG data
    augmented = ecg_data + noise
    
    return augmented

class ECGAugmentor:
    def __init__(self):
        self.ecg_lead_order = [
            'I', 'II', 'III', 'aVR', 'aVF', 'aVL',
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6'
        ]
    
    def time_warp(self, signal, factor = 0.25, points = 4):
        return time_warp_ecg(signal, w = factor, m = points)
    
    def permutation(self, signal, m=4):
        return permutation_augmentation(signal, m=m)
    
    def add_noise(self, signal, sigma = 0.2):
        return add_gaussian_noise(signal, sigma=sigma)
    
    def jitter(self, ecg_data, noise_factor=0.03):
        noise = np.random.randn(*ecg_data.shape) * noise_factor
        augmented = ecg_data + noise
        return np.clip(augmented, -1.0, 1.0)
    
    def random_scaling(self, signal, scale_range=(0.8, 2.1)):
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return signal * scale_factor
    
    def flip_signal(self, signal):
        if np.random.rand() > 0.5:
            return -signal
        return signal
    
    def no_aug(self, signal):
        return signal

    def ecg2vcg(self, ecg):

        return ecg

    def vcg_random_rotate(self, vcg):
        return 0
    
    def vcg_random_scale(self, vcg):
        return 0
    
    def vcg2ecg(self, vcg):
        return vcg

    
    def ecg_random_mask(self, ecg):
        return ecg
    
    def randomAugment(self, signal, augmentations):
        selected_augments = random.sample(augmentations, 1)
        for augment in selected_augments:
            signal = augment(signal)
        return signal
    

    def augment(self, signal):
        signal = self.randomAugment(
            signal,
            augmentations = [
                self.time_warp,
                self.permutation,
                # self.add_noise,
                self.jitter,
                self.no_aug
            ]
        )
        return signal

class VCG_ECGAugmentor_OLD:
    """
    Source
    3KG: Contrastive Learning of 12-Lead Electrocardiograms using
    Physiologically-Inspired Augmentations. Gopal et al. ML4H 2021.

    https://github.com/Jwoo5/fairseq-signals/blob/master/
    fairseq_signals/data/ecg/perturb_ecg_dataset.py
    """

    def __init__(self, cfg):
        self.angle = cfg.cromotex.vcg_aug_angle
        self.scale = cfg.cromotex.vcg_aug_scale
        self.mask_ratio = cfg.cromotex.ecg_aug_mask_ratio
    
    def _get_other_four_leads(self, I, II):
        """
        calculate other four leads (III, aVR, aVL, aVF)
        from the first two leads (I, II)
        """
        III = -I + II
        aVR = -(I + II) / 2
        aVL = I - II/2
        aVF = -I/2 + II
        return III, aVR, aVL, aVF

    def augment(self, ecg, mode='crossmodal'):
        """
        mode: 'crossmodal' or 'unimodal'
        if crossmodal, only a single augmented ECG is returned
        if unimodal, two augmented ECGs from same source are returned
        """
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.numpy()
        leads_taken = [0,1,6,7,8,9,10,11]
        other_leads = [2,3,4,5]
        ecg = ecg[leads_taken]

        D_i = np.array(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ]
        )
        D = np.linalg.pinv(D_i)

        vcg = D_i @ ecg

        if self.angle:
            angles = np.random.uniform(-self.angle, self.angle, size=6)
            R1 = R.from_euler('zyx', angles[:3], degrees=True)
            R2 = R.from_euler('zyx', angles[3:], degrees=True)
            if hasattr(R1, "as_dcm"): #old versions of scipy
                R1 = R1.as_dcm()
                R2 = R2.as_dcm()
            else:
                R1 = R1.as_matrix()
                R2 = R2.as_matrix()
        else:
            R1 = np.diag((1,1,1))
            R2 = np.diag((1,1,1))
        
        if self.scale:
            scales = np.random.uniform(1, self.scale, size=6)
            S1 = np.diag(scales[:3])
            S2 = np.diag(scales[3:])
        else:
            S1 = np.diag((1,1,1))
            S2 = np.diag((1,1,1))
        
        res1 = D @ S1 @ R1 @ vcg
        res2 = D @ S2 @ R2 @ vcg

        sample_size = ecg.shape[-1]

        ecg1 = np.zeros((12, sample_size))
        ecg2 = np.zeros((12, sample_size))

        ecg1[leads_taken] = res1
        ecg1[other_leads] = self._get_other_four_leads(res1[0], res1[1])

        ecg2[leads_taken] = res2
        ecg2[other_leads] = self._get_other_four_leads(res2[0], res2[1])

        if self.mask_ratio:
            sample_size = ecg.shape[-1]
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = np.random.randint(0, sample_size, size=24)
            end_indices = np.array(
                [
                    s + offset if s + offset <= sample_size else sample_size
                    for s in start_indices
                ]
            )
            leftovers = np.array(
                [
                    s + offset - sample_size if s + offset > sample_size else 0
                    for s in start_indices
                ]
            )

            for i in range(12):
                ecg1[i, start_indices[i]:end_indices[i]] = 0
                ecg1[i, 0:leftovers[i]] = 0
            
                ecg2[i, start_indices[i+12]:end_indices[i+12]] = 0
                ecg2[i, 0:leftovers[i+12]] = 0
        
        ecg1 = torch.from_numpy(ecg1)
        ecg2 = torch.from_numpy(ecg2)
        if mode == 'crossmodal':
            return ecg1
        return ecg1, ecg2

class VCG_ECGAugmentor:
    """
    Source
    3KG: Contrastive Learning of 12-Lead Electrocardiograms using
    Physiologically-Inspired Augmentations. Gopal et al. ML4H 2021.

    https://github.com/Jwoo5/fairseq-signals/blob/master/
    fairseq_signals/data/ecg/perturb_ecg_dataset.py
    """

    def __init__(self, cfg=None):
        if cfg is None:
            self.angle = 45
            self.scale = 1.5
            self.mask_ratio = 0.25
        else:
            self.angle = cfg.cromotex.vcg_aug_angle
            self.scale = cfg.cromotex.vcg_aug_scale
            self.mask_ratio = cfg.cromotex.ecg_aug_mask_ratio
    
    def _get_other_four_leads(self, I, II):
        """
        calculate other four leads (III, aVR, aVL, aVF)
        from the first two leads (I, II)
        """
        III = -I + II
        aVR = -(I + II) / 2
        aVL = I - II/2
        aVF = -I/2 + II
        return III, aVR, aVL, aVF

    def augment(self, ecg):
        """
        mode: 'crossmodal' or 'unimodal'
        if crossmodal, only a single augmented ECG is returned
        if unimodal, two augmented ECGs from same source are returned
        """
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.numpy()
        leads_taken = [0,1,6,7,8,9,10,11]
        other_leads = [2,3,4,5]
        ecg = ecg[leads_taken]

        D_i = np.array(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ]
        )
        D = np.linalg.pinv(D_i)

        vcg = D_i @ ecg

        if self.angle:
            angles = np.random.uniform(-self.angle, self.angle, size=3)
            R1 = R.from_euler('zyx', angles, degrees=True)
            if hasattr(R1, "as_dcm"): #old versions of scipy
                R1 = R1.as_dcm()
            else:
                R1 = R1.as_matrix()
        else:
            R1 = np.diag((1,1,1))
        
        if self.scale:
            scales = np.random.uniform(1, self.scale, size=3)
            S1 = np.diag(scales)
        else:
            S1 = np.diag((1,1,1))
        
        res1 = D @ S1 @ R1 @ vcg

        sample_size = ecg.shape[-1]

        ecg1 = np.zeros((12, sample_size))

        ecg1[leads_taken] = res1
        ecg1[other_leads] = self._get_other_four_leads(res1[0], res1[1])

        if self.mask_ratio:
            sample_size = ecg.shape[-1]
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = np.random.randint(0, sample_size, size=12)
            end_indices = np.array(
                [
                    s + offset if s + offset <= sample_size else sample_size
                    for s in start_indices
                ]
            )
            leftovers = np.array(
                [
                    s + offset - sample_size if s + offset > sample_size else 0
                    for s in start_indices
                ]
            )

            for i in range(12):
                ecg1[i, start_indices[i]:end_indices[i]] = 0
                ecg1[i, 0:leftovers[i]] = 0
            
        
        ecg1 = torch.from_numpy(ecg1)
        return ecg1

    def augment_double(self, ecg):
        """
        mode: 'crossmodal' or 'unimodal'
        if crossmodal, only a single augmented ECG is returned
        if unimodal, two augmented ECGs from same source are returned
        """
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.numpy()
        leads_taken = [0,1,6,7,8,9,10,11]
        other_leads = [2,3,4,5]
        ecg = ecg[leads_taken]

        D_i = np.array(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ]
        )
        D = np.linalg.pinv(D_i)

        vcg = D_i @ ecg

        if self.angle:
            angles = np.random.uniform(-self.angle, self.angle, size=6)
            R1 = R.from_euler('zyx', angles[:3], degrees=True)
            R2 = R.from_euler('zyx', angles[3:], degrees=True)
            if hasattr(R1, "as_dcm"): #old versions of scipy
                R1 = R1.as_dcm()
                R2 = R2.as_dcm()
            else:
                R1 = R1.as_matrix()
                R2 = R2.as_matrix()
        else:
            R1 = np.diag((1,1,1))
            R2 = np.diag((1,1,1))
        
        if self.scale:
            scales = np.random.uniform(1, self.scale, size=6)
            S1 = np.diag(scales[:3])
            S2 = np.diag(scales[3:])
        else:
            S1 = np.diag((1,1,1))
            S2 = np.diag((1,1,1))
        
        res1 = D @ S1 @ R1 @ vcg
        res2 = D @ S2 @ R2 @ vcg

        sample_size = ecg.shape[-1]

        ecg1 = np.zeros((12, sample_size))
        ecg2 = np.zeros((12, sample_size))

        ecg1[leads_taken] = res1
        ecg1[other_leads] = self._get_other_four_leads(res1[0], res1[1])

        ecg2[leads_taken] = res2
        ecg2[other_leads] = self._get_other_four_leads(res2[0], res2[1])

        if self.mask_ratio:
            sample_size = ecg.shape[-1]
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = np.random.randint(0, sample_size, size=24)
            end_indices = np.array(
                [
                    s + offset if s + offset <= sample_size else sample_size
                    for s in start_indices
                ]
            )
            leftovers = np.array(
                [
                    s + offset - sample_size if s + offset > sample_size else 0
                    for s in start_indices
                ]
            )

            for i in range(12):
                ecg1[i, start_indices[i]:end_indices[i]] = 0
                ecg1[i, 0:leftovers[i]] = 0
            
                ecg2[i, start_indices[i+12]:end_indices[i+12]] = 0
                ecg2[i, 0:leftovers[i+12]] = 0
        
        ecg1 = torch.from_numpy(ecg1).float()
        ecg2 = torch.from_numpy(ecg2).float()
        return ecg1, ecg2

    def augment_torch(self, ecg):
        ### This doesn't work!
        if not isinstance(ecg, torch.Tensor):
            raise TypeError("Expected input as a torch.Tensor")

        leads_taken = torch.tensor(
            [0, 1, 6, 7, 8, 9, 10, 11], device=ecg.device
        )
        other_leads = torch.tensor([2, 3, 4, 5], device=ecg.device)

        ecg_selected = ecg[leads_taken]

        # D_i and D as PyTorch tensors
        D_i = torch.tensor(
            [
                [0.156, -0.009, -0.172, -0.073, 0.122, 0.231, 0.239, 0.193],
                [-0.227, 0.886, 0.057, -0.019, -0.106, -0.022, 0.040, 0.048],
                [0.021, 0.102, -0.228, -0.310, -0.245, -0.063, 0.054, 0.108]
            ], device=ecg.device, dtype=torch.float32
        )

        D = torch.linalg.pinv(D_i)

        vcg = torch.matmul(D_i, ecg_selected)  # Shape: [3, time]

        # Rotation Matrix in PyTorch (ZYX Euler Angles)
        if self.angle:
            angles = torch.empty(
                3, device=ecg.device
            ).uniform_(
                -self.angle, self.angle
            )
            cos_a = torch.cos(torch.deg2rad(angles))
            sin_a = torch.sin(torch.deg2rad(angles))

            R_z = torch.tensor(
                [[cos_a[0], -sin_a[0], 0], [sin_a[0], cos_a[0], 0], [0, 0, 1]],
                device=ecg.device
            )
            R_y = torch.tensor(
                [[cos_a[1], 0, sin_a[1]], [0, 1, 0], [-sin_a[1], 0, cos_a[1]]],
                device=ecg.device
            )
            R_x = torch.tensor(
                [[1, 0, 0], [0, cos_a[2], -sin_a[2]], [0, sin_a[2], cos_a[2]]],
                device=ecg.device
            )

            R1 = R_z @ R_y @ R_x 
        else:
            R1 = torch.eye(3, device=ecg.device)

        if self.scale:
            scales = torch.empty(3, device=ecg.device).uniform_(1, self.scale)
            S1 = torch.diag(scales)
        else:
            S1 = torch.eye(3, device=ecg.device)

        res1 = torch.matmul(D, torch.matmul(S1, torch.matmul(R1, vcg)))

        sample_size = ecg.shape[-1]

        ecg1 = torch.zeros(
            (12, sample_size), device=ecg.device, dtype=torch.float32
        )

        ecg1[leads_taken] = res1
        ecg1[other_leads] = self._get_other_four_leads(res1[0], res1[1])

        if self.mask_ratio:
            offset = math.floor(sample_size * self.mask_ratio)

            start_indices = torch.randint(
                0, sample_size, (12,), device=ecg.device
            )
            end_indices = (start_indices + offset) % sample_size

            for i in range(12):
                if start_indices[i] < end_indices[i]:  
                    ecg1[i, start_indices[i]:end_indices[i]] = 0
                else:  # Wrap case: split into two zeroed parts
                    ecg1[i, start_indices[i]:] = 0
                    ecg1[i, :end_indices[i]] = 0
        return ecg1