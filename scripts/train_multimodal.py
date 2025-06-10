import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
import logging
from rich.logging import RichHandler
from rich.progress import track
import hydra
from omegaconf import DictConfig
import mlflow
import time

from cromotex.models.cromotex import get_cromotex
from cromotex.models.ahnp_loss import AHNPLoss
import cromotex.utils.metrics as metrics
import cromotex.utils.utils as utils
from cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched
from cromotex.utils.datasets import CXR_ECG_MatchedDataset

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

MODEL_NAME = 'cromotex'

def train_one_epoch(
    cfg,
    model,
    dataloader,
    criterion,
    optimizer,
    epoch,
    device,
    **kwargs
):
    model.train()
    running_loss = 0.0

    lr = lr_sched(cfg.cromotex_train.optim, epoch)
    
    optimizer = model.set_lr(cfg, optimizer, lr)

    optimizer.zero_grad()

    for idx, (img, ecg, labels) in enumerate(dataloader):
        img, ecg, labels = img.to(device), ecg.to(device), labels.to(device)
        
        img_proj, ts_proj, ts_logits = model(img, ecg)
        loss = criterion(img_proj, ts_proj, ts_logits, labels)
        loss = loss / cfg.cromotex_train.optim.grad_accum_steps

        loss.backward()

        if cfg.cromotex_train.optim.grad_clip > 0.0:
            grad_clip = cfg.cromotex_train.optim.grad_clip
            if cfg.cromotex_train.data_parallel:
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), grad_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.cromotex_train.optim.grad_accum_steps
        
        if (idx + 1) % cfg.cromotex_train.optim.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            mlflow.log_metric(
                "loss_batch", loss.item(), step=idx + len(dataloader) * epoch
            )
            
    loss_epoch = running_loss / len(dataloader)
    train_info = {}
    train_info['loss'] = loss_epoch
    train_info['epoch'] = epoch

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
        for idx, (img, ecg, labels) in enumerate(dataloader):
            img, ecg = img.to(device), ecg.to(device)
            labels = labels.to(device)

            img_proj, ts_proj, ts_logits = model(img, ecg)

            loss = criterion(img_proj, ts_proj, ts_logits, labels)

            running_loss += loss.item()
            preds = (torch.sigmoid(ts_logits) > 0.5).float()
            probs = torch.sigmoid(ts_logits)
    
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_pred_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
    
    accuracy = correct_predictions / total_samples
    loss_epoch = running_loss / len(dataloader)

    all_labels = torch.cat(all_labels, dim=0).unsqueeze(1)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)

    auroc_scores = metrics.auroc(all_labels, all_pred_probs)
    auprc_scores = metrics.auprc(all_labels, all_pred_probs)

    val_info = {}
    val_info['loss'] = loss_epoch
    val_info['accuracy'] = accuracy
    val_info['auroc'] = auroc_scores
    val_info['auprc'] = auprc_scores

    mlflow.log_metric("loss_val", loss_epoch, step=epoch)    
    mlflow.log_metric(
        f"auroc", auroc_scores[0], step=epoch
    )
    mlflow.log_metric(
        f"prauc", auprc_scores[0], step=epoch
    )
    return val_info

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config"
)
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.cromotex_train.seed)
    torch.manual_seed(cfg.cromotex_train.seed)
    torch.cuda.manual_seed(cfg.cromotex_train.seed)
    torch.cuda.manual_seed_all(cfg.cromotex_train.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if cfg.cromotex_train.data_parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", cfg.cromotex_train.gpu_id)

    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.cromotex_train.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.cromotex_train.mlflow_expt_name
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
    
    model = get_cromotex(cfg)
    if cfg.cromotex_train.data_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if isinstance(model, torch.nn.DataParallel):
        img_augs_train, img_augs_val, ts_augmentor = (
            model.module.get_augmentations()
        )
    else:
        img_augs_train, img_augs_val, ts_augmentor = model.get_augmentations()

    train_data = CXR_ECG_MatchedDataset(
        cfg,
        'datasets/processed/train_matched.h5',
        img_augs_train, ts_augmentor
    )

    val_data = CXR_ECG_MatchedDataset(
        cfg,
        'datasets/processed/val_matched.h5',
        img_augs_val, None
    )

    train_labels = train_data.get_labels()
    pos_ratio = 0.25
    neg_ratio = 1 - pos_ratio
    class_weights = {0: 1.0 / neg_ratio, 1: 1.0 / pos_ratio}

    sample_weights = torch.tensor(
        [class_weights[label.item()] for label in train_labels]
    )
    
    generator = torch.Generator().manual_seed(cfg.cromotex_train.seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels),
        replacement=True, generator=generator
    )

    def seed_worker(worker_id):
        np.random.seed(cfg.cromotex_train.seed + worker_id)
        torch.manual_seed(cfg.cromotex_train.seed + worker_id)

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.cromotex_train.batch_size,
        sampler=sampler,
        # shuffle=True,
        num_workers=cfg.cromotex_train.num_dataloader_workers,
        worker_init_fn=seed_worker
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.cromotex_train.batch_size,
        shuffle=False,
        num_workers=cfg.cromotex_train.num_dataloader_workers
    )

    criterion = AHNPLoss(cfg)

    if cfg.cromotex.img_encoder_freeze:
        if isinstance(model, torch.nn.DataParallel):
            for param in model.module.image_encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.image_encoder.parameters():
                param.requires_grad = False

    if isinstance(model, torch.nn.DataParallel):
        optimizer = model.module.get_optimizer(cfg, model, criterion)
    else:
        optimizer = model.get_optimizer(cfg, model)#, criterion)

    start_epoch = 0
    if cfg.cromotex_train.resume_from_last_ckpt:
        checkpoint_data = utils.load_train_checkpoint(
            f'cromotex_best_{cfg.pathology}.pth', model, optimizer
        )
        model, optimizer, last_epoch, mlflow_run_id = checkpoint_data
        start_epoch = last_epoch + 1
        mlflow.start_run(
            run_id=mlflow_run_id, experiment_id=experiment.experiment_id
        )
        logger.info(f"Loaded checkpoint @ epoch {epoch}")
    else:
        tags = {'mlflow.note.content': cfg.cromotex_train.mlflow_run_notes}
        mlflow.start_run(experiment_id=experiment.experiment_id, tags=tags)

    if isinstance(model, torch.nn.DataParallel):
        total_params = sum(
            p.numel() for p in model.module.parameters() if p.requires_grad
        )
    else:
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
    best_auroc = 0.0
    best_prauc = 0.0
    train_infos = []
    val_infos = []
    for epoch in range(start_epoch, start_epoch + cfg.cromotex_train.num_epochs):
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
        
        run_name = mlflow.active_run().data.tags.get('mlflow.runName')
        # Save best val_loss checkpoint
        if val_info['loss'] < best_val_loss:
            best_val_loss = val_info['loss']
            if cfg.cromotex_train.save_ckpt:
                utils.save_checkpoint(
                    f'{MODEL_NAME}_best_loss_{cfg.pathology}_{run_name}.pth',
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        if val_info['auroc'][list(val_info['auroc'].keys())[0]] > best_auroc:
            best_auroc = val_info['auroc'][list(val_info['auroc'].keys())[0]]
            if cfg.cromotex_train.save_ckpt:
                utils.save_checkpoint(
                    f'{MODEL_NAME}_best_auroc_{cfg.pathology}_{run_name}.pth',
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        if val_info['auprc'][list(val_info['auprc'].keys())[0]] > best_prauc:
            best_prauc = val_info['auprc'][list(val_info['auprc'].keys())[0]]
            if cfg.cromotex_train.save_ckpt:
                utils.save_checkpoint(
                    f'{MODEL_NAME}_best_prauc_{cfg.pathology}_{run_name}.pth',
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        mlflow.log_metric("loss_best_val", best_val_loss, step=epoch)
        mlflow.log_metric("auroc_best_val", best_auroc, step=epoch)
        mlflow.log_metric("prauc_best_val", best_prauc, step=epoch)

        # Save last checkpoint    
        if cfg.cromotex_train.save_ckpt:
            utils.save_checkpoint(
                f'{MODEL_NAME}_last_{cfg.pathology}_{run_name}.pth',
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.cromotex_train.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=3
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

if __name__ == '__main__':
    main()