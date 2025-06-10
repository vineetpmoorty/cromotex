from torch.utils.data import Dataset
import torch
from torchvision import transforms
import h5py
import os
import hydra

DATASET_BASE_DIR = '/home/shared/cromotex/' #or '' if local to current project

pathologies = [
    "cardiomegaly", "edema",
    "enlarged_cardiomediastinum", "pleural_effusion", "pneumonia"
]

class CXRPretrainDataset(Dataset):
    def __init__(self, cfg, hdf5_file_path, augmentations=None):
        hdf5_file_path = hydra.utils.to_absolute_path(
            os.path.join(
                DATASET_BASE_DIR, hdf5_file_path
            )
        )
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.images = self.hdf5_file['images']
        self.labels = self.hdf5_file['labels'][:]
        self.augmentations = augmentations
        pathology = cfg.pathology
        self.pathology_index = pathologies.index(pathology)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = label[self.pathology_index]
        # Apply augmentations if specified
        if self.augmentations:
            img = transforms.ToPILImage()(img)  # Convert to PIL image
            img = self.augmentations(img)  # Apply augmentations
            #Ensure augmentations has toTensor() at the end
        return img, label

    def get_labels(self):
        return self.labels[:, self.pathology_index]

class ECGPretrainDataset(Dataset):
    def __init__(self, hdf5_file_path, augmentations=None):
        hdf5_file_path = hydra.utils.to_absolute_path(
            os.path.join(
                DATASET_BASE_DIR, hdf5_file_path
            )
        )
        print(f"XXXXXX {hdf5_file_path}")
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.ecg = self.hdf5_file['ecg']
        self.augmentations = augmentations

    def __len__(self):
        return len(self.ecg)
    
    def __getitem__(self, idx):
        ecg = torch.tensor(self.ecg[idx], dtype=torch.float32)
        if self.augmentations:
            ecg = self.augmentations(ecg)
        return ecg

class CXR_ECG_MatchedDataset(Dataset):
    def __init__(
        self, cfg, hdf5_file_path, cxr_augmentations=None, ecg_augmentor=None
    ):  
        hdf5_file_path = hydra.utils.to_absolute_path(
            os.path.join(
                DATASET_BASE_DIR, hdf5_file_path
            )
        )
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.images = self.hdf5_file['images']
        self.ecg = self.hdf5_file['ecg']
        self.labels = self.hdf5_file['labels'][:]
        self.cxr_augmentations = cxr_augmentations
        self.ecg_augmentor = ecg_augmentor
        pathology = cfg.pathology
        self.pathology_index = pathologies.index(pathology)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        label = label[self.pathology_index]

        if self.cxr_augmentations:
            img = transforms.ToPILImage()(img)  # Convert to PIL image
            img = self.cxr_augmentations(img)  # Apply augmentations
            #Ensure augmentations has toTensor() at the end

        ecg = torch.tensor(self.ecg[idx], dtype=torch.float32)

        if self.ecg_augmentor:
            ecg = torch.tensor(
                self.ecg_augmentor.augment(ecg),
                dtype=torch.float32
            )

        return img, ecg, label

    def get_labels(self):
        labels = self.labels[:, self.pathology_index]
        return labels