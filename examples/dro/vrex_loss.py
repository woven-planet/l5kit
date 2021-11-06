import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VRexLossComputer(nn.Module):
    def __init__(self, n_groups, group_counts, group_str, device, logger=None,
                 penalty_weight=1.0):
        super(VRexLossComputer, self).__init__()
        self.is_robust = True
        self.n_groups = n_groups
        self.group_counts = group_counts.to(device)
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = group_str
        self.device = device
        # self.logger = logger
        self.time_steps = 0
        self.penalty_weight = penalty_weight

    def loss(self, per_sample_losses, group_idx):
        per_sample_losses = per_sample_losses.mean(dim=1)
        # compute per-group losses
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)
        # Variance
        rex_penalty = torch.sum((group_loss - group_loss.unsqueeze(1))**2) / (self.n_groups ** 2)

        # compute overall loss
        erm_loss = per_sample_losses.mean()
        loss = erm_loss
        if self.is_robust:
            loss += self.penalty_weight * rex_penalty

        # log
        # self.log_group_weights(group_loss, group_count, rex_penalty, erm_loss)

        return loss

    @torch.jit.unused
    def log_group_weights(self, group_loss, group_count, rex_penalty, erm_loss):
        if self.logger is not None:
            self.logger.record(f'group_loss/erm', erm_loss.item())
            self.logger.record(f'group_loss/vrex', rex_penalty.item())
            self.time_steps += 1
            for idx, g_name in enumerate(self.group_str):
                self.logger.record(f'group_loss/{g_name}', group_loss[idx].item())
                self.logger.record(f'group_count/{g_name}', group_count[idx].item())
                self.logger.dump(self.time_steps)

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count
