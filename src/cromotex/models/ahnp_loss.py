import torch
import torch.nn as nn
import torch.nn.functional as F

class UnimodalUnsupConLoss(nn.Module):
    def __init__(self, cfg):
        super(UnimodalUnsupConLoss, self).__init__()
        self.cfg = cfg
        self.temp = cfg.pretrain_ecg.temperature
    
    def forward(self, proj):
        batch_size = proj.shape[0]
        proj1 = proj[:batch_size//2]
        proj2 = proj[batch_size//2:]
        loss1 = self._unimodal_unsup_con_loss(proj1, proj2, self.temp)
        loss2 = self._unimodal_unsup_con_loss(proj2, proj1, self.temp)
        return 0.5*(loss1 + loss2)

    def _unimodal_unsup_con_loss(self, proj1, proj2, temp=0.1):
        batch_size = proj1.shape[0]
        proj_dim = proj1.shape[1]    
        proj1 = F.normalize(proj1, dim=-1)  # [batch_size, proj_dim]
        proj2 = F.normalize(proj2, dim=-1)  # [batch_size, proj_dim]

        all_dot_prods_diff_view = torch.mm(proj1, proj2.T) / temp
        all_dot_prods_diff_view_stable = (
            all_dot_prods_diff_view
            - torch.max(all_dot_prods_diff_view, dim=1, keepdim=True)[0]
        )
        all_exp_dot_prods_diff_view = torch.exp(all_dot_prods_diff_view_stable)
        sum_all_exp_dot_prods_diff_view = all_exp_dot_prods_diff_view.sum(
            dim=-1
        )

        exp_pos_dot_prods = all_exp_dot_prods_diff_view.diagonal()

        all_dot_prods_same_view = torch.mm(proj1, proj1.T) / temp
        all_dot_prods_same_view_stable = (
            all_dot_prods_same_view
            - torch.max(all_dot_prods_same_view, dim=1, keepdim=True)[0]
        )
        
        all_exp_dot_prods_same_view = torch.exp(all_dot_prods_same_view_stable).clone()
        all_exp_dot_prods_same_view.fill_diagonal_(0)
        sum_all_exp_dot_prods_same_view = all_exp_dot_prods_same_view.sum(
            dim=-1
        )

        sum_exp_dot_prods_both_views = (
            sum_all_exp_dot_prods_diff_view + sum_all_exp_dot_prods_same_view
        )

        loss = -torch.mean(
            torch.log(exp_pos_dot_prods / sum_exp_dot_prods_both_views)
        )
        return loss

class AHNPLoss(nn.Module):
    def __init__(self, cfg):
        """
        AHNP Loss
        """
        super(AHNPLoss, self).__init__()

        self.temperature = cfg.cromotex_train.temperature
        self.lambda_cross_contrast = cfg.cromotex_train.lambda_cross_contrast
        self.lambda_uni_contrast = cfg.cromotex_train.lambda_uni_contrast
        self.lambda_classif = cfg.cromotex_train.lambda_classif
        self.cfg = cfg
        self.hard_neg_weights = cfg.cromotex_train.hard_neg_weights
        try:
            if cfg.cromotex.learnable_loss_weights:
                self.w = nn.Parameter(
                    torch.tensor(torch.logit(torch.tensor(0.5)))
                )
            else:
                self.w = None
        except:
            self.w = None

        try:
            self.dir_pos_pair_weight = (
                cfg.cromotex.direct_positive_pair_weight
            )
        except:
            self.dir_pos_pair_weight = 1.0

    def forward(self, img_proj, ts_proj, ts_logits, labels):
        """
        Compute the AHNP loss.

        :param img_proj: shape [batch_size, proj_dim], image projections.
        :param ts_proj: shape [batch_size, proj_dim], time-series projections.
        :param labels: shape [batch_size], ground truth labels.
        :return: torch.Tensor, scalar hybrid loss.
        """
        if self.cfg.cromotex_train.hard_neg_weights == 'none':
            cross_modal_sup_con_loss1 = self._crossmodal_loss(
                img_proj, ts_proj, labels, self.temperature, unsupervised=False
            )
            cross_modal_sup_con_loss2 = self._crossmodal_loss(
                ts_proj, img_proj, labels, self.temperature, unsupervised=False
            )
            cross_modal_sup_con_loss = (
                0.5*(cross_modal_sup_con_loss1 + cross_modal_sup_con_loss2)
            )
        else:
            cross_modal_sup_con_loss1 = self._crossmodal_loss_hard_neg(
                img_proj, ts_proj, labels, self.temperature, unsupervised=False
            )
            cross_modal_sup_con_loss2 = self._crossmodal_loss_hard_neg(
                ts_proj, img_proj, labels, self.temperature, unsupervised=False
            )
            cross_modal_sup_con_loss = (
                0.5*(cross_modal_sup_con_loss1 + cross_modal_sup_con_loss2)
            )

        ahnp_loss = (
            self.lambda_cross_contrast * cross_modal_sup_con_loss
        )
        return ahnp_loss

    def _crossmodal_loss(
        self, img_proj, ts_proj, labels, temp=0.1, unsupervised=False
    ):
        """
        Cross-modal supervised contrastive loss.
        img_proj, ts_proj: shape[batch_size, proj_dim]
        labels: shape[batch_size]
        """
        batch_size = img_proj.shape[0]
        proj_dim = img_proj.shape[1]

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1) # [batch_size, 1]
        
        img_proj = F.normalize(img_proj, dim=-1)  # [batch_size, proj_dim]
        ts_proj = F.normalize(ts_proj, dim=-1)    # [batch_size, proj_dim]

        dot_prods = torch.mm(img_proj, ts_proj.T) # [batch_size, batch_size]
        dot_prods = dot_prods / temp

        dot_prods_stable = dot_prods - torch.max(dot_prods, dim=1, keepdim=True)[0]
        exp_dot_prods = torch.exp(dot_prods_stable)
        
        if not unsupervised:
            label_mask = (labels == labels.T).float()
        else:
            label_mask = torch.eye(batch_size).to(img_proj.device)
        
        all_dot_prods = torch.sum(exp_dot_prods, dim=1) 
        all_dot_prods_tiled = all_dot_prods.unsqueeze(1).repeat(1, batch_size)
        all_dot_prods_tiled = all_dot_prods_tiled + 1e-6 #ensure it is not zero

        log_term = torch.log(exp_dot_prods/all_dot_prods_tiled)

        label_mask.fill_diagonal_(self.dir_pos_pair_weight)
        
        sum_log_term = torch.sum(log_term * label_mask, dim=1)

        normalized_sum_log_term = sum_log_term / torch.sum(label_mask, dim=1)

        loss = -torch.mean(normalized_sum_log_term)
        return loss

    def _crossmodal_loss_hard_neg(
        self, img_proj, ts_proj, labels, temp=0.1, unsupervised=False
    ):
        """
        Cross-modal supervised contrastive loss.
        Hard negatives are weighed more heavily.
        img_proj, ts_proj: shape[batch_size, proj_dim]
        labels: shape[batch_size]
        """
        batch_size = img_proj.shape[0]
        proj_dim = img_proj.shape[1]

        if len(labels.shape) == 1:
            labels = labels.unsqueeze(1) # [batch_size, 1]
        
        img_proj = F.normalize(img_proj, dim=-1)  # [batch_size, proj_dim]
        ts_proj = F.normalize(ts_proj, dim=-1)    # [batch_size, proj_dim]

        dot_prods = torch.mm(img_proj, ts_proj.T) # [batch_size, batch_size]
        dot_prods = dot_prods / temp

        dot_prods_stable = (
            dot_prods - torch.max(dot_prods, dim=1, keepdim=True)[0]
        )
        exp_dot_prods = torch.exp(dot_prods_stable)
        
        if not unsupervised:
            label_mask = (labels == labels.T).float()
        else:
            label_mask = torch.eye(batch_size).to(img_proj.device)
        # label_mask = (labels == labels.T).float()
        
        alpha = self.cfg.cromotex_train.hard_neg_alpha

        if self.hard_neg_weights == 'linear':
            max_sim = torch.max(
                dot_prods * (1 - label_mask), dim=-1, keepdim=True
            )[0]
            min_sim = torch.min(
                dot_prods * (1 - label_mask), dim=-1, keepdim=True
            )[0]

            factors = (
                1 + (
                    ((alpha - 1)/(max_sim - min_sim + 1e-8))
                    * (dot_prods - min_sim)
                )
            )
            
            all_dot_prods = torch.sum(
                exp_dot_prods * factors, dim=1
            )
        elif self.hard_neg_weights == 'topk':
            k = int(
                self.cfg.cromotex_train.hard_neg_topk_fraction * batch_size
            )
            topk_indices = torch.topk(exp_dot_prods, k, dim=1).indices
            factors = torch.ones_like(exp_dot_prods, dtype=torch.float32)
            factors.scatter_(1, topk_indices, alpha)
            
            all_dot_prods = torch.sum(
                exp_dot_prods * factors, dim=1
            )
        elif self.hard_neg_weights == 'exp':
            factors = 1 + torch.exp(alpha * dot_prods)
            
            all_dot_prods = torch.sum(
                exp_dot_prods * factors, dim=1
            )

        dir_pos_pair_weight = 3.0

        all_dot_prods_tiled = all_dot_prods.unsqueeze(1).repeat(1, batch_size)
        all_dot_prods_tiled = all_dot_prods_tiled + 1e-6 #ensure it is not zero

        log_term = torch.log(1e-8 + (exp_dot_prods/all_dot_prods_tiled))

        label_mask.fill_diagonal_(dir_pos_pair_weight)
        
        sum_log_term = torch.sum(log_term * label_mask, dim=1)

        normalized_sum_log_term = sum_log_term / torch.sum(label_mask, dim=1)

        loss = -torch.mean(normalized_sum_log_term)
        return loss
        