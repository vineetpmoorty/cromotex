import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import hydra
import time

from cromotex.models.image_encoder import get_image_encoder
from cromotex.models.timeseries_encoder import ECGPatchTransformer
from cromotex.utils.ts_augmentations import ECGAugmentor, VCG_ECGAugmentor
from cromotex.utils.utils import load_train_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPClassifHead(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifHead, self).__init__()
        embed_dim = cfg.cromotex.embed_dim
        hidden_dim = cfg.cromotex.classif_head_hid_dim
        num_layers = cfg.cromotex.classif_head_num_layers
        dropout = cfg.cromotex.classif_head_dropout

        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim)) 
                layers.append(nn.GELU()) 
                layers.append(nn.Dropout(dropout))
                
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class CroMoTEXFinetune(nn.Module):
    def __init__(self, cfg, logger):
        super(CroMoTEXFinetune, self).__init__()
        self.cfg = cfg
        cromotex = CroMoTEXPatchTransformer(cfg)
        
        ckpt_filename = cfg.finetune.ckpt_filename
        if len(ckpt_filename) > 0:
            self.cromotex, _, _, _ = load_train_checkpoint(
                ckpt_filename, cromotex
            )
        
            logger.info(
                f"Loaded trained checkpoint {cfg.finetune.ckpt_filename}"
            )
    
            ckpt_file = os.path.join(
                hydra.utils.to_absolute_path('checkpoints'),
                cfg.finetune.ckpt_filename
            )
            ckpt_time = time.ctime(os.path.getmtime(ckpt_file))
            logger.info(f"Checkpoint was saved at: {ckpt_time}")
        else:
            self.cromotex = cromotex
            logger.info("No checkpoint loaded. Training from scratch.")

        self.classif_head = MLPClassifHead(cfg)

        self.ts_augmentor = ECGAugmentor()
        img_augs = get_image_encoder(cfg).get_augmentations(cfg)
        self.img_augs_train, self.img_augs_val = img_augs
    
    def get_augmentations(self):
        return self.img_augs_train, self.img_augs_val, self.ts_augmentor
    
    def forward(self, ts):
        ts_embeds, _ = self.cromotex(None, ts, True)
        logits = self.classif_head(ts_embeds)
        return logits, ts_embeds

    def get_optimizer(self, cfg, model, loss=None):

        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        param_groups = [
            {
                'params': m.classif_head.parameters(),
                'lr': cfg.finetune.optim.lr_peak,
                'weight_decay': cfg.finetune.optim.weight_decay,
                'name': 'classif_head'
            },
        ]
        if not cfg.finetune.freeze_backbone:
            param_groups.append(
                {
                    'params': m.cromotex.timeseries_encoder.parameters(),
                    'lr': cfg.finetune.optim.lr_peak,
                    'weight_decay': cfg.finetune.optim.weight_decay,
                    'name': 'ts_encoder'
                },
            )
        optimizer = optim.AdamW(param_groups)
        return optimizer
    
    def set_lr(self, cfg, optimizer, lr):
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'ts_encoder':
                param_group['lr'] = lr
            elif param_group['name'] == 'classif_head':
                param_group['lr'] = lr
        return optimizer

class CroMoTEXPatchTransformer(nn.Module):
    """
    First transform ECG using a series of Conv1D layers 
    to get ECG embeddings of shape [batch_size, embed_dim, num_patches].
    Then apply a transformer to get embeddings of shape [batch_size, embed_dim]

    Note: Performed the best out of all models considered so far.
    Best auroc 0.72 and best auprc 0.23 on effusion.
    """
    def __init__(self, cfg):
        super(CroMoTEXPatchTransformer, self).__init__()
        self.cfg = cfg
        self.image_encoder = get_image_encoder(cfg).densenet.features

        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'pretrain_img_best_{cfg.pathology}.pth'
        )

        checkpoint = torch.load(filepath, map_location='cpu')
        encoder_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('densenet.features')
        }

        if isinstance(self.image_encoder, torch.nn.DataParallel):
            self.image_encoder.module.load_state_dict(
                encoder_state_dict, strict=False
            )
        else:
            self.image_encoder.load_state_dict(
                encoder_state_dict, strict=False
            )

        self.timeseries_encoder = ECGPatchTransformer(cfg)

        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'pretrain_ecg_best.pth'
        )

        checkpoint = torch.load(filepath, map_location='cpu')
        ts_encoder_state_dict = {
            k.replace('timeseries_encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('timeseries_encoder')
        }

        self.timeseries_encoder.load_state_dict(ts_encoder_state_dict)
        
        self.img_feature_dim = get_image_encoder(cfg).feature_dim

        self.img_proj_linear = nn.Linear(
            self.img_feature_dim, cfg.cromotex.proj_dim
        )
        self.ts_proj_linear = nn.Linear(
            self.timeseries_encoder.classifier.in_features,
            cfg.cromotex.proj_dim
        )

        if self.cfg.cromotex.ecg_augments == 'basic_augments':
            self.ts_augmentor = ECGAugmentor()
        else:
            self.ts_augmentor = VCG_ECGAugmentor(cfg)

        img_augs = get_image_encoder(cfg).get_augmentations(cfg)
        self.img_augs_train, self.img_augs_val = img_augs

    def forward(self, images, ts, finetune=False):
        ts_embeds, ts_logits = self.timeseries_encoder(ts)
        ts_embeds = F.normalize(ts_embeds, dim=-1)

        if finetune:
            return ts_embeds, ts_logits

        ts_proj = self.ts_proj_linear(ts_embeds)
        ts_proj = F.normalize(ts_proj, dim=-1)

        batch_size = images.size(0)

        images = images.mean(1).unsqueeze(1)
        img_feats = self.image_encoder(images)
        # below transforms are from 
        # github/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py
        img_embeds = F.relu(img_feats, inplace=True)
        img_embeds = F.adaptive_avg_pool2d(
            img_embeds, (1, 1)).view(batch_size, -1
        )
        # img_embeds = F.normalize(img_embeds, dim=-1)
        img_proj = self.img_proj_linear(img_embeds)
        img_proj = F.normalize(img_proj, dim=-1)

        return img_proj, ts_proj, ts_logits

    def get_augmentations(self):
        return self.img_augs_train, self.img_augs_val, self.ts_augmentor

    def get_optimizer(self, cfg, model, loss=None):

        if cfg.cromotex.learnable_loss_weights and loss is None:
            raise ValueError(
                'Loss object must be provided when'
                'using learnable loss weights'
            )
        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        param_groups = [
            {
                'params': m.img_proj_linear.parameters(),
                'lr': (
                    cfg.cromotex_train.optim.lr_peak * 
                    cfg.cromotex.proj_linear_lr_scaler
                ),
                'weight_decay': cfg.cromotex_train.optim.weight_decay,
                'name': 'img_proj_linear'
            },
            {
                'params': m.timeseries_encoder.parameters(),
                'lr': cfg.cromotex_train.optim.lr_peak,
                'weight_decay': cfg.cromotex_train.optim.weight_decay,
                'name': 'timeseries_encoder'
            },
            {
                'params': m.ts_proj_linear.parameters(),
                'lr': (
                    cfg.cromotex_train.optim.lr_peak 
                    * cfg.cromotex.proj_linear_lr_scaler
                ),
                'weight_decay': cfg.cromotex_train.optim.weight_decay,
                'name': 'ts_proj_linear'
            },
        ]
        if not cfg.cromotex.img_encoder_freeze:
            param_groups.append(
                {
                    'params': m.image_encoder.parameters(),
                    'lr': (
                        cfg.cromotex_train.optim.lr_peak
                        * cfg.cromotex.img_encoder_lr_scaler
                    ),
                    'weight_decay': cfg.cromotex_train.optim.weight_decay,
                    'name': 'img_encoder'
                }
            )
        optimizer = optim.AdamW(param_groups)
        return optimizer

    def set_lr(self, cfg, optimizer, lr):
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'img_proj_linear':
                param_group['lr'] = (
                    lr * cfg.cromotex.proj_linear_lr_scaler
                )
            elif param_group['name'] == 'ts_proj_linear':
                param_group['lr'] = (
                    lr * cfg.cromotex.proj_linear_lr_scaler
                )
            elif param_group['name'] == 'timeseries_encoder': 
                param_group['lr'] = lr
            elif param_group['name'] == 'loss_weights':
                param_group['lr'] = (
                    lr * cfg.cromotex.learnable_weights_lr_scaler
                )
            elif param_group['name'] == 'img_encoder':
                param_group['lr'] = (
                    lr * cfg.cromotex.img_encoder_lr_scaler
                )

        return optimizer

class CroMoTEXECGPretrain(nn.Module):
    """
    First transform ECG using a series of Conv1D layers 
    to get ECG embeddings of shape [batch_size, embed_dim, num_patches].
    Then apply a transformer to get embeddings of shape [batch_size, embed_dim]
    """
    def __init__(self, cfg):
        super(CroMoTEXECGPretrain, self).__init__()
        self.cfg = cfg
        
        self.timeseries_encoder = ECGPatchTransformer(cfg)
        self.ts_proj_linear = nn.Linear(
            self.timeseries_encoder.classifier.in_features,
            cfg.cromotex.proj_dim
        )

        if self.cfg.cromotex.ecg_augments == 'basic_augments':
            self.ts_augmentor = ECGAugmentor()
        else:
            self.ts_augmentor = VCG_ECGAugmentor(cfg)

    def forward(self, ts):
        ts_embeds, ts_logits = self.timeseries_encoder(ts)
        ts_embeds = F.normalize(ts_embeds, dim=-1)

        ts_proj = self.ts_proj_linear(ts_embeds)
        ts_proj = F.normalize(ts_proj, dim=-1)
        return ts_proj, ts_embeds, ts_logits

    def get_augmentations(self):
        return self.ts_augmentor

    def get_optimizer(self, cfg, model, loss=None):

        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        param_groups = [
            {
                'params': m.timeseries_encoder.parameters(),
                'lr': cfg.cromotex_train.optim.lr_peak,
                'weight_decay': cfg.cromotex_train.optim.weight_decay,
                'name': 'timeseries_encoder'
            },
            {
                'params': m.ts_proj_linear.parameters(),
                'lr': (
                    cfg.cromotex_train.optim.lr_peak 
                    * cfg.cromotex.proj_linear_lr_scaler
                ),
                'weight_decay': cfg.cromotex_train.optim.weight_decay,
                'name': 'ts_proj_linear'
            },
        ]
        optimizer = optim.AdamW(param_groups)
        return optimizer

    def set_lr(self, cfg, optimizer, lr):
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'ts_proj_linear':
                param_group['lr'] = (
                    lr * cfg.cromotex.proj_linear_lr_scaler
                )
            elif param_group['name'] == 'timeseries_encoder': 
                param_group['lr'] = lr
        return optimizer

def get_cromotex(cfg):
    if cfg.cromotex.name == 'cromotex_patch_transformer':
        model = CroMoTEXPatchTransformer(cfg)
    else:
        raise ValueError(
            f'Invalid cromotex name'
        )

    return model
    
