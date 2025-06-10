# code layout
# import and configure model
# make the dataloader class
# perform data augmentations for cxr data
# define the training loop
# define the evaluation loop
# define the main function
# add logging and save checkpoints

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

from cromolts.models.cromolts import CroMoLTSFinetune
from cromolts.models.cmscc_loss import CMSCCLoss
import cromolts.utils.metrics as metrics
import cromolts.utils.utils as utils
from cromolts.utils.utils import lr_linear_rise_cosine_decay as lr_sched
from cromolts.utils.datasets import CXR_ECG_MatchedDataset

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
    running_loss_unimodal = 0.0
    running_loss_multimodal = 0.0
    running_loss_classif = 0.0

    lr = lr_sched(cfg.finetune.optim, epoch)
    
    if isinstance(model, torch.nn.DataParallel):
        optimizer = model.module.set_lr(cfg, optimizer, lr)
    else:
        optimizer = model.set_lr(cfg, optimizer, lr)

    optimizer.zero_grad()

    # for images, labels in track(dataloader, description=f"Epoch {epoch}"):
    for idx, (img, ecg, labels) in enumerate(dataloader):
        # if idx > 5:
        #     break #testing
        ecg, labels = ecg.to(device), labels.to(device)
        labels = labels.unsqueeze(-1).float()
        # ts_logits = model(None, ecg, True) #BIOT, ViT/ECCL
        ts_logits = model(ecg) #Cromolts
        loss = criterion(ts_logits, labels)
        loss = loss / cfg.finetune.optim.grad_accum_steps

        loss.backward()

        if cfg.finetune.optim.grad_clip > 0.0:
            grad_clip = cfg.finetune.optim.grad_clip
            if cfg.finetune.data_parallel:
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), grad_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.finetune.optim.grad_accum_steps

        ###
        total_norm = 0
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        parameters = [
            p for p in m.parameters()
            if p.grad is not None and p.requires_grad
        ]
        
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        mlflow.log_metric(
            "grad_norm_batch", total_norm, step=idx + len(dataloader) * epoch
        )

        if (idx + 1) % cfg.finetune.optim.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            mlflow.log_metric(
                "loss_batch", loss.item(), step=idx + len(dataloader) * epoch
            )
    
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
        for idx, (img, ecg, labels) in enumerate(dataloader):
            # if idx > 5:
            #     break #testing
            ecg = ecg.to(device)
            labels = labels.to(device)
            labels = labels.unsqueeze(-1).float()

            # ts_logits = model(None, ecg, True) #BIOT
            ts_logits = model(ecg) #Cromolts

            loss = criterion(ts_logits, labels)
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

    mlflow.log_metric(
        f"auroc", auroc_scores[0], step=epoch
    )
    mlflow.log_metric(
        f"prauc", auprc_scores[0], step=epoch
    )
    mlflow.log_metric("f1", f1_score, step=epoch)
    return val_info

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config"
)
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.finetune.seed)
    torch.manual_seed(cfg.finetune.seed)
    torch.cuda.manual_seed(cfg.finetune.seed)
    torch.cuda.manual_seed_all(cfg.finetune.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if cfg.finetune.data_parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", cfg.finetune.gpu_id)

    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.finetune.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.finetune.mlflow_expt_name
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
    
    model = CroMoLTSFinetune(cfg, logger)
    # model = CrossModalDropFuse(cfg)
    # model = CroMoLTSViT(cfg)

    # filepath = os.path.join(
    # hydra.utils.to_absolute_path('checkpoints'),
    #     'dropfuse_best_loss_edema_calm-seal-707.pth'
    # )

    # checkpoint = torch.load(filepath, map_location='cpu')
    # encoder_state_dict = checkpoint['model_state_dict']
    # model.load_state_dict(encoder_state_dict, strict=False)

    if cfg.finetune.data_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    img_augs_train, img_augs_val, ts_augmentor = m.get_augmentations()
    
    for param in m.parameters():
        #Freeze entire model
        param.requires_grad = False
    if not cfg.finetune.freeze_backbone:
        for param in m.cromolts.timeseries_encoder.parameters():
        # for param in m.timeseries_encoder.parameters():
            #Only un-freeze the ts encoder
            try:
                param.requires_grad = True
            except:
                pass
    # for param in m.classif_head.parameters():
    #     #Only un-freeze the ts encoder
    #     param.requires_grad = True

    optimizer = m.get_optimizer(cfg, model)

    train_data = CXR_ECG_MatchedDataset(
        cfg,
        'datasets/processed/train_matched.h5',
        None, ts_augmentor
    )

    val_data = CXR_ECG_MatchedDataset(
        cfg,
        'datasets/processed/test_matched.h5',
        None, None
    )

    train_labels = train_data.get_labels()
    pos_ratio = 0.2
    neg_ratio = 1 - pos_ratio
    class_weights = {0: 1.0 / neg_ratio, 1: 1.0 / pos_ratio}

    sample_weights = torch.tensor(
        [class_weights[label.item()] for label in train_labels]
    )
    
    generator = torch.Generator().manual_seed(cfg.cromolts_train.seed)
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels),
        replacement=True, generator=generator
    )

    def seed_worker(worker_id):
        np.random.seed(cfg.cromolts_train.seed + worker_id)
        torch.manual_seed(cfg.cromolts_train.seed + worker_id)

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.finetune.batch_size,
        sampler=sampler,
        worker_init_fn=seed_worker,
        # shuffle=True,
        num_workers=cfg.finetune.num_dataloader_workers
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.finetune.num_dataloader_workers
    )

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0

    mlflow.start_run(experiment_id=experiment.experiment_id)

    if isinstance(model, torch.nn.DataParallel):
        total_params = sum(
            p.numel() for p in model.module.parameters() if p.requires_grad
        )
    else:
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    logger.info(f"Model size: {(total_params)} parameters")

    current_run_id = mlflow.active_run().info.run_id
    current_run_name = mlflow.active_run().info.run_name
    logger.info(f"Starting from epoch {start_epoch}")
    logger.info(f"mlflow run name: [bold red]{current_run_name}")
    mlflow.log_params(utils.format_cfg(cfg))

    ckpt_run_name = cfg.finetune.ckpt_filename.split('.')[0].split('_')[-1]

    best_val_loss = float('inf')
    best_auroc = 0.0
    best_prauc = 0.0
    best_f1 = 0.0
    train_infos = []
    val_infos = []
    for epoch in range(start_epoch, start_epoch + cfg.finetune.num_epochs):
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
            if cfg.finetune.save_ckpt:
                fname = f'cromolts_finetuned_best_loss_{cfg.pathology}'
                fname += f'_{ckpt_run_name}.pth'
                utils.save_checkpoint(
                    fname,
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        if val_info['auroc'][list(val_info['auroc'].keys())[0]] > best_auroc:
            best_auroc = val_info['auroc'][list(val_info['auroc'].keys())[0]]
            if cfg.finetune.save_ckpt:
                fname = f'cromolts_finetuned_best_auroc_{cfg.pathology}'
                fname += f'_{ckpt_run_name}.pth'
                utils.save_checkpoint(
                    fname,
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        if val_info['auprc'][list(val_info['auprc'].keys())[0]] > best_prauc:
            best_prauc = val_info['auprc'][list(val_info['auprc'].keys())[0]]
            if cfg.finetune.save_ckpt:
                fname = f'cromolts_finetuned_best_prauc_{cfg.pathology}'
                fname += f'_{ckpt_run_name}.pth'
                utils.save_checkpoint(
                    fname,
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        if val_info['f1'] > best_f1:
            best_f1 = val_info['f1']
            if cfg.finetune.save_ckpt:
                fname = f'cromolts_finetuned_best_f1_{cfg.pathology}'
                fname += f'_{ckpt_run_name}.pth'
                utils.save_checkpoint(
                    fname,
                    model,
                    optimizer,
                    epoch,
                    current_run_id
                )
        # Save last checkpoint    
        if cfg.finetune.save_ckpt:
            fname = f'cromolts_finetuned_last_{cfg.pathology}'
            fname += f'_{ckpt_run_name}.pth'
            utils.save_checkpoint(
                fname,
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.finetune.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=3
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

if __name__ == '__main__':
    main()