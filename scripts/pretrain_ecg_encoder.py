import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import logging
from rich.logging import RichHandler
from rich.progress import track
import hydra
from omegaconf import DictConfig
import mlflow
import time

from cromotex.models.cromotex import CroMoTEXECGPretrain
import cromotex.utils.metrics as metrics
import cromotex.utils.utils as utils
from cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched
from cromotex.utils.datasets import ECGPretrainDataset
from cromotex.models.ahnp_loss import UnimodalUnsupConLoss

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

def train_one_epoch(
    cfg,
    model,
    dataloader,
    criterion,
    optimizer,
    ecg_augments,
    epoch,
    device,
):
    model.train()
    running_loss = 0.0

    lr = lr_sched(cfg.pretrain_ecg.optim, epoch)
    optimizer = model.set_lr(cfg, optimizer, lr)
    optimizer.zero_grad()

    for idx, ecg in enumerate(dataloader):
        augs1 = []
        augs2 = []
        for i in range(ecg.shape[0]):
            e1, e2 = ecg_augments.augment_double(ecg[i])
            augs1.append(e1)
            augs2.append(e2)
        
        ecg1 = torch.stack(augs1, dim=0)
        ecg2 = torch.stack(augs2, dim=0)
        ecg = torch.cat([ecg1, ecg2], dim=0)

        ecg = ecg.to(device)

        outputs, _, _ = model(ecg)
        loss = criterion(outputs)
        loss = loss / cfg.pretrain_ecg.optim.grad_accum_steps

        loss.backward()

        if cfg.pretrain_ecg.optim.grad_clip > 0.0:
            grad_clip = cfg.pretrain_ecg.optim.grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.pretrain_ecg.optim.grad_accum_steps

        if (idx + 1) % cfg.pretrain_ecg.optim.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            mlflow.log_metric(
                "loss_batch", loss.item(), step=idx + len(dataloader) * epoch
            )

    loss_epoch = running_loss / len(dataloader)
    train_info = {}
    train_info['loss'] = loss_epoch
    train_info['epoch'] = epoch

    utils.log_train_info_to_mlflow(train_info)
    return train_info

def evaluate(cfg, model, dataloader, criterion, ecg_augments, epoch, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for idx, ecg in enumerate(dataloader):
            augs1 = []
            augs2 = []
            for i in range(ecg.shape[0]):
                e1, e2 = ecg_augments.augment_double(ecg[i])
                augs1.append(e1)
                augs2.append(e2)
            
            ecg1 = torch.stack(augs1, dim=0)
            ecg2 = torch.stack(augs2, dim=0)
            ecg = torch.cat([ecg1, ecg2], dim=0)

            ecg = ecg.to(device)
            outputs, _, _ = model(ecg)
            loss = criterion(outputs)
            running_loss += loss.item()
            
    loss_epoch = running_loss / len(dataloader)
    val_info = {}
    val_info['loss'] = loss_epoch
    mlflow.log_metric("loss_val", loss_epoch, step=epoch)
    return val_info

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda", cfg.pretrain_ecg.gpu_id)
    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.pretrain_ecg.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.pretrain_ecg.mlflow_expt_name
    )

    logger = logging.getLogger("mlflow")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    rich_handler = RichHandler(
        show_level=False,
        show_time=True,
        show_path=False,
        markup=True
    )
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%H:%M:%S]"
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)
    
    model = CroMoTEXECGPretrain(cfg)
    model.to(device)

    ecg_augments = model.get_augmentations()

    train_data = ECGPretrainDataset(        
        'datasets/processed/pretrain_ecg_train.h5',
        augmentations=None
    )

    val_data = ECGPretrainDataset(
        'datasets/processed/pretrain_ecg_val.h5',
        augmentations=None
    )

    half_batch_size = cfg.pretrain_ecg.batch_size // 2

    train_dataloader = DataLoader(
        train_data,
        batch_size=half_batch_size,
        shuffle=True,
        num_workers=cfg.pretrain_ecg.num_dataloader_workers
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=half_batch_size,
        shuffle=False,
        num_workers=cfg.pretrain_ecg.num_dataloader_workers
    )

    criterion = UnimodalUnsupConLoss(cfg)
    
    optimizer = model.get_optimizer(cfg, model)

    start_epoch = 0
    if cfg.pretrain_ecg.resume_from_last_ckpt:
        checkpoint_data = utils.load_train_checkpoint(
            f'pretrain_ecg_last_{cfg.pathology}.pth', model, optimizer
        )
        model, optimizer, last_epoch, mlflow_run_id = checkpoint_data
        start_epoch = last_epoch + 1
        mlflow.start_run(
            run_id=mlflow_run_id, experiment_id=experiment.experiment_id
        )
        logger.info(f"Loaded checkpoint @ epoch {epoch}")
    else:
        mlflow.start_run(experiment_id=experiment.experiment_id)

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logger.info(f"Model size: {(total_params/1e6):.2f}M parameters")

    current_run_id = mlflow.active_run().info.run_id
    current_run_name = mlflow.active_run().info.run_name
    logger.info(f"Starting from epoch {start_epoch}")
    logger.info(f"mlflow run name: [bold red]{current_run_name}")
    mlflow.log_params(utils.format_cfg(cfg))

    best_val_loss = float('inf')
    train_infos = []
    val_infos = []
    for epoch in range(start_epoch, start_epoch + cfg.pretrain_ecg.num_epochs):
        start_time = time.time()

        train_info = train_one_epoch(
            cfg,
            model,
            train_dataloader,
            criterion,
            optimizer,
            ecg_augments,
            epoch,
            device
        )
    
        val_info = evaluate(
            cfg,
            model,
            val_dataloader,
            criterion,
            ecg_augments,
            epoch,
            device
        )

        train_infos.append(train_info)
        val_infos.append(val_info)

        epoch_time = time.time() - start_time
        logger.info(utils.log_epoch_metrics(train_info, val_info, epoch_time))
        
        # Save best checkpoint
        if val_info['loss'] < best_val_loss:
            best_val_loss = val_info['loss']
            
            if cfg.pretrain_ecg.save_ckpt:
                utils.save_checkpoint(
                    f'biot_pretrain_ecg_best.pth',
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        
        # Save last checkpoint    
        if cfg.pretrain_ecg.save_ckpt:
            utils.save_checkpoint(
                f'biot_pretrain_ecg_last.pth',
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.pretrain_ecg.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=5
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

if __name__ == '__main__':
    main()