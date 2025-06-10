import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import math
import torch
from torch.utils.data import WeightedRandomSampler
import random
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from typing import List

def save_checkpoint(filename, model, optim, epoch, mlflow_run_id):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': (
            model.module.state_dict() if isinstance(
                model, torch.nn.DataParallel
            )
            else model.state_dict()
        ),
        'optimizer_state_dict': optim.state_dict(),
        'mlflow_run_id': mlflow_run_id,
    }
    filepath = hydra.utils.to_absolute_path('checkpoints')
    Path(filepath).mkdir(parents=True, exist_ok=True)    

    filepath = os.path.join(
        hydra.utils.to_absolute_path('checkpoints'),
        filename
    )
    torch.save(save_dict, filepath)

def load_train_checkpoint(filename, model, optim=None):
    filepath = os.path.join(
        hydra.utils.to_absolute_path('checkpoints'),
        filename
    )
    checkpoint = torch.load(filepath, map_location='cpu')
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    mlflow_run_id = checkpoint['mlflow_run_id']
    return model, optim, epoch, mlflow_run_id

def load_inference_checkpoint(filename, model):
    filepath = os.path.join(
        hydra.utils.to_absolute_path('checkpoints'),
        filename
    )
    checkpoint = torch.load(filepath)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    return model

def format_cfg(cfg: DictConfig):
    d = OmegaConf.to_container(cfg)

    def flatten_dict(d, parent_key='', sep='.'):
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items
    d = flatten_dict(d)
    return d

def format_epoch_time(start, end):
    seconds = end - start
    if seconds > 60 and seconds < 60*60:
        time = str(round(seconds/60, 2))
        unit = ' min'
    elif seconds >= 60*60:
        time = str(round(seconds/3600, 2))
        unit = ' hr'
    else:
        time = str(round(seconds, 2))
        unit = ' s'
    return time + unit

def lr_linear_rise_cosine_decay(optim_cfg, epoch):
    """
    custom learning rate scheduler from nanoGPT
    Linear warm-up from 0 to lr > Cosine decay down to min_lr
    """
    start_lr = optim_cfg.lr_peak/10.0
    learning_rate = optim_cfg.lr_peak
    min_lr = optim_cfg.lr_peak/10.0
    warmup_iters = optim_cfg.warmup_epochs
    lr_decay_iters = optim_cfg.decay_epochs
    
    # 1) linear warmup for warmup_iters steps
    if epoch < warmup_iters:
        return start_lr + (learning_rate - start_lr) * epoch / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if epoch > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (epoch - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr) 

def get_weighted_sampler(labels: np.ndarray):
    num_samples = len(labels)
    positive_counts = labels.sum(axis=0)  # Sum positives per label
    negative_count = num_samples - np.any(labels, axis=1).sum() 

    # Calculate per-class weights
    label_frequencies = positive_counts / num_samples
    negative_frequency = negative_count / num_samples

    # Inverse weight for sampling (add epsilon to avoid division by zero)
    class_weights = 1 / (label_frequencies + 1e-6)
    negative_weight = 1 / (negative_frequency + 1e-6)

    sample_weights = []
    for l in labels:
        if np.any(l == 1):
            sample_weight = np.max(class_weights[l == 1])
        else:
            sample_weight = negative_weight
            
        sample_weights.append(sample_weight)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler

def log_epoch_metrics(train_info: dict, val_info: dict, epoch_time) -> str:
    log_str_1 = (
        f"ep {str(train_info['epoch']).zfill(3)} | "
        f"trn loss: {train_info['loss']:.4f} | "
        f"val loss: {val_info['loss']:.4f} | "
    )
    if 'auroc' in val_info:
        log_str2 = (
            f"val auroc: {', '.join([f'{val_info['auroc'][k]:.2f}' for k in val_info['auroc']])} | "
        )
    else:
        log_str2 = ""
    if 'auprc' in val_info:
        log_str3 = (
            f"val auprc: {', '.join([f'{val_info['auprc'][k]:.2f}' for k in val_info['auprc']])} | "
        )
    else:
        log_str3 = ""
    log_str4 = f"time: {format_epoch_time(0, epoch_time)}"
    return log_str_1 + log_str2 + log_str3 + log_str4

def log_train_info_to_mlflow(train_info: dict):
    mlflow.log_metric(
        "loss_train", train_info['loss'], step=train_info['epoch']
    )

def log_random_images_to_mlflow(dataloader, epoch):
    random_batch = next(iter(dataloader))
    images, labels = random_batch  

    assert len(images) >= 4, "Batch size must be at least 4 to plot 2x2 grid."

    random_indices = random.sample(range(len(images)), 4)
    random_images = images[random_indices] 
    random_labels = labels[random_indices]

    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle(f"Random Images - Epoch {epoch}", fontsize=12)

    for i, ax in enumerate(axs.flat):
        image = random_images[i]
        image = (image - image.min()) / (image.max() - image.min())
        label = ", ".join(
            random_labels[i].cpu().numpy().astype(int).astype(str).tolist()
        )

        if image.ndim == 3:
            image = image.permute(1, 2, 0).numpy()

        ax.imshow(image, cmap="gray" if image.shape[-1] == 1 else None)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plot_filename = f"random_images_epoch_{epoch}.png"
    mlflow.log_figure(fig, plot_filename)
    plt.close(fig)
    
def early_stop(
    train_infos: List[dict],
    val_infos: List[dict],
    patience: int = 5,
    delta: float = 1e-6,
) -> bool:
    # Stop if last `patience` val losses are strictly
    # increasing by more than `delta`
    if len(val_infos) < patience:
        return False
    else:
        val_losses = [val_info['loss'] for val_info in val_infos[-patience:]]
        if all(
            (
                (val_losses[i+1] - val_losses[i]) > 
                delta for i in range(patience - 1)
            )
        ):
            return True

        return False
