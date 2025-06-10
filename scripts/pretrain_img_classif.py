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

from cromotex.models.image_encoder import get_image_encoder
import cromotex.utils.metrics as metrics
import cromotex.utils.utils as utils
from cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched
from cromotex.utils.datasets import CXRPretrainDataset

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

def train_one_epoch(
    cfg,
    model,
    dataloader,
    criterion,
    optimizer,
    epoch,
    device
):
    model.train()
    running_loss = 0.0

    lr = lr_sched(cfg.pretrain_img.optim, epoch)

    for param_group in optimizer.param_groups:
        if param_group['name'] == 'backbone':
            param_group['lr'] = lr * cfg.pretrain_img.optim.backbone_lr_scaler
        elif param_group['name'] == 'classifier': 
            param_group['lr'] = lr

    optimizer.zero_grad()

    # for images, labels in track(dataloader, description=f"Epoch {epoch}"):
    for idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / cfg.pretrain_img.optim.grad_accum_steps

        loss.backward()

        if cfg.pretrain_img.optim.grad_clip > 0.0:
            grad_clip = cfg.pretrain_img.optim.grad_clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.pretrain_img.optim.grad_accum_steps

        if (idx + 1) % cfg.pretrain_img.optim.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            mlflow.log_metric(
                "loss_batch", loss.item(), step=idx + len(dataloader) * epoch
            )
    
    utils.log_random_images_to_mlflow(dataloader, epoch)

    loss_epoch = running_loss / len(dataloader)
    train_info = {}
    train_info['loss'] = loss_epoch
    train_info['epoch'] = epoch

    for param_group in optimizer.param_groups:
        train_info[f"lr_{param_group['name']}"] = param_group['lr']

    utils.log_train_info_to_mlflow(train_info)
    return train_info

def evaluate(cfg, model, dataloader, criterion, epoch, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = [] #batch ground truths
    all_preds = [] #batch predictions
    all_pred_probs = [] #batch prediction probabilities

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            probs = torch.sigmoid(outputs)
    
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_pred_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
    
    accuracy = correct_predictions / total_samples
    loss_epoch = running_loss / len(dataloader)

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)

    auroc_scores = metrics.auroc(all_labels, all_pred_probs)
    auprc_scores = metrics.auprc(all_labels, all_pred_probs)
    f1_score = metrics.f1(all_labels, all_pred_probs)

    val_info = {}
    val_info['loss'] = loss_epoch
    val_info['accuracy'] = accuracy
    val_info['auroc'] = auroc_scores
    val_info['auprc'] = auprc_scores
    val_info['f1'] = f1_score

    metrics.log_precision_recall_curves_to_mlflow(
        all_labels, all_pred_probs, epoch
    )
    
    mlflow.log_metric("loss_val", loss_epoch, step=epoch)
    mlflow.log_metric("accuracy_val", accuracy, step=epoch)
    mlflow.log_metric("f1", f1_score, step=epoch)
    for label_idx in range(all_labels.shape[1]):
        mlflow.log_metric(
            f"auroc_val_{label_idx}", auroc_scores[label_idx], step=epoch
        )
        mlflow.log_metric(
            f"auprc_val_{label_idx}", auprc_scores[label_idx], step=epoch
        )
    return val_info

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda", cfg.pretrain_img.gpu_id)

    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.pretrain_img.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.pretrain_img.mlflow_expt_name
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
    
    model = get_image_encoder(cfg)

    model.to(device)

    train_augmentations, val_augmentations = model.get_augmentations(cfg)

    train_data = CXRPretrainDataset(
        cfg,
        'datasets/processed/pretrain_train.h5',
        augmentations=train_augmentations
    )

    val_data = CXRPretrainDataset(
        cfg,
        'datasets/processed/test_matched.h5',
        augmentations=val_augmentations
    )

    if cfg.pretrain_img.weighted_sampling:
        sampler = utils.get_weighted_sampler(train_data.get_labels())
        shuffle = False
    else:
        shuffle = True
        sampler = None

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.pretrain_img.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.pretrain_img.num_dataloader_workers
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.pretrain_img.batch_size,
        shuffle=False,
        num_workers=cfg.pretrain_img.num_dataloader_workers
    )

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam([
        {
            'params': model.densenet.features.parameters(),
            'lr': (
                cfg.pretrain_img.optim.lr_peak * 
                cfg.pretrain_img.optim.backbone_lr_scaler
            ),
            'weight_decay': cfg.pretrain_img.optim.weight_decay,
            'name': 'backbone'
        },
        {
            'params': model.densenet.classifier.parameters(),
            'lr': cfg.pretrain_img.optim.lr_peak,
            'weight_decay': cfg.pretrain_img.optim.weight_decay,
            'name': 'classifier'
        }
    ])

    start_epoch = 0
    if cfg.pretrain_img.resume_from_last_ckpt:
        checkpoint_data = utils.load_train_checkpoint(
            'pretrain_img_last_{cfg.pathology}.pth', model, optimizer
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
    for epoch in range(start_epoch, start_epoch + cfg.pretrain_img.num_epochs):
        start_time = time.time()

        train_info = train_one_epoch(
            cfg,
            model,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            device
        )
        # train_info = {'epoch': epoch, 'loss': 0} #Eval only
        val_info = evaluate(
            cfg,
            model,
            val_dataloader,
            criterion,
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
            
            if cfg.pretrain_img.save_ckpt:
                utils.save_checkpoint(
                    f'pretrain_img_best_{cfg.pathology}.pth',
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        
        # Save last checkpoint    
        if cfg.pretrain_img.save_ckpt:
            utils.save_checkpoint(
                f'pretrain_img_last_{cfg.pathology}.pth',
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.pretrain_img.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=4
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

if __name__ == '__main__':
    main()