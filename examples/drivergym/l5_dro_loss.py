import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossComputer(nn.Module):
    def __init__(self, n_groups, group_counts, group_str, device, alpha=None, gamma=0.1,
                 adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False, btl=False):
        super(LossComputer, self).__init__()
        self.is_robust = True
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = n_groups
        self.group_counts = group_counts.to(device)
        self.group_frac = self.group_counts/self.group_counts.sum()
        self.group_str = group_str

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(device)

        if self.is_robust and btl:
            assert alpha, 'alpha must be specified'

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(device)

        self.device = device

    def loss(self, per_sample_losses, group_idx):
        per_sample_losses = per_sample_losses.mean(dim=1)
        # compute per-group losses
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.is_robust and not self.btl:
            actual_loss, weights = self.compute_robust_loss(group_loss, group_count)
        # elif self.is_robust and self.btl:
        #      actual_loss, weights = self.compute_robust_loss_btl(group_loss, group_count)
        else:
            actual_loss = per_sample_losses.mean()
            weights = None

        return actual_loss

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    # def compute_robust_loss_btl(self, group_loss, group_count):
    #     adjusted_loss = self.exp_avg_loss + self.adj/torch.sqrt(self.group_counts)
    #     return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    # def compute_robust_loss_greedy(self, group_loss, ref_loss):
    #     sorted_idx = ref_loss.sort(descending=True)[1]
    #     sorted_loss = group_loss[sorted_idx]
    #     sorted_frac = self.group_frac[sorted_idx]

    #     mask = torch.cumsum(sorted_frac, dim=0)<=self.alpha
    #     weights = mask.float() * sorted_frac /self.alpha
    #     last_idx = mask.sum()
    #     weights[last_idx] = 1 - weights.sum()
    #     weights = sorted_frac*self.min_var_weight + weights*(1-self.min_var_weight)

    #     robust_loss = sorted_loss @ weights

    #     # sort the weights back
    #     _, unsort_idx = sorted_idx.sort()
    #     unsorted_weights = weights[unsort_idx]
    #     return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(self.device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma*(group_count>0).float()) * (self.exp_avg_initialized>0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss*curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized>0) + (group_count>0)
