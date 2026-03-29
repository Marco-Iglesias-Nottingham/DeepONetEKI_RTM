import torch
import torch.nn.functional as F

from emulator_tools.data_utils import add_positional_channels
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, f1_score
import numpy as np
import os
import json
from scipy.io import savemat, loadmat
import torch.distributed as dist
import torch.distributed as dist
import scipy.sparse as sp
import pickle
from collections import defaultdict



def loss_pressure(
    p_pred, p_true):
        
    err2_std = (p_pred - p_true).pow(2)
    return    err2_std.mean()


def create_bin_index_map(space_coords, k, device):
    coords = space_coords.T.to(device)  # shape: (n_space, 2)
    coords_min = coords.min(dim=0, keepdim=True)[0]
    coords_max = coords.max(dim=0, keepdim=True)[0]
    normalized_coords = (coords - coords_min) / (coords_max - coords_min + 1e-6)

    grid_size = int(k ** 0.5)
    if grid_size ** 2 < k:
        grid_size += 1

    x_bins = torch.linspace(0, 1, grid_size + 1, device=device)
    y_bins = torch.linspace(0, 1, grid_size + 1, device=device)

    x_idx = torch.bucketize(normalized_coords[:, 0].contiguous(), x_bins) - 1
    y_idx = torch.bucketize(normalized_coords[:, 1].contiguous(), y_bins) - 1
    bin_ids = (x_idx * grid_size + y_idx).tolist()

    bin_to_indices = defaultdict(list)
    for idx, bin_id in enumerate(bin_ids):
        bin_to_indices[bin_id].append(idx)

    return bin_to_indices, list(bin_to_indices.keys())

def sample_k_diverse_indices(bin_to_indices, available_bins, k, device):
    rand_bins = torch.randperm(len(available_bins), device=device)
    selected_indices = []
    for bin_i in rand_bins:
        bin_id = available_bins[bin_i.item()]
        candidates = bin_to_indices[bin_id]
        if len(candidates) > 0:
            rand_idx = torch.randint(0, len(candidates), (1,), device=device).item()
            selected_indices.append(candidates[rand_idx])
        if len(selected_indices) >= k:
            break
    return torch.tensor(selected_indices[:k], device=device)




def train(
    model, dataloader, optimizer, device, epoch, trunk_inputs_encoded,
    p_mean, p_std,
    num_times=15, k=50, world_size=1,
    lumped_probs: torch.Tensor | None = None,
    bce_weight: float = 0.1
):
    """
    Importance sample spatial nodes ~ Categorical(lumped_probs).
    Time steps are uniform. Everything stays normalized.
    Pressure losses are scaled by σ²; BCE unchanged.
    """
    model.train()
    n_space, n_time = trunk_inputs_encoded.shape[:2]
    assert lumped_probs is not None and lumped_probs.numel() == n_space, \
        "Provide lumped_probs over spatial nodes (sum==1)."

    trunk_enc_dev = trunk_inputs_encoded.to(device)

    tot = tot_p  = tot_f = 0.0
    n_batches = 0


    for branch_batch, scalar_batch, target_batch in dataloader:
        branch_batch = branch_batch.to(device)
        scalar_batch = scalar_batch.to(device)
        target_batch = target_batch.to(device)

        B = branch_batch.size(0)
        target_batch = target_batch.view(B, n_space, n_time, target_batch.shape[-1])

        # --- sample uniform times (num_times per sample)
        time_indices = torch.stack(
            [torch.randperm(n_time, device=device)[:num_times] for _ in range(B)],
            dim=0
        )
        time_indices, _ = torch.sort(time_indices, dim=1)

        # --- allocate sampled sets
        P = num_times * k
        trunk_sub  = torch.empty((B, P, trunk_enc_dev.shape[-1]), device=device)
        target_sub = torch.empty((B, P, target_batch.shape[-1]), device=device)

        # --- importance-sample space by lumped mass
        for i in range(B):
            ptr = 0
            for j in range(num_times):
                t_idx = time_indices[i, j]
                s_idx = torch.multinomial(lumped_probs, k, replacement=True)  # [k]
                trunk_sub[i, ptr:ptr+k]  = trunk_enc_dev[s_idx, t_idx, :]
                target_sub[i, ptr:ptr+k] = target_batch[i, s_idx, t_idx, :]
                ptr += k

        branch_batch = add_positional_channels(branch_batch)

        optimizer.zero_grad()
        p_pred, f_pred = model(branch_batch, scalar_batch, trunk_sub)  # [B,P,1] each

        p_true = target_sub[..., 0:1]
        f_true = target_sub[..., 1:2]

        loss_p = loss_pressure(p_pred, p_true)
        f = torch.sigmoid(f_pred)
        loss_f = F.mse_loss(f, f_true, reduction='mean')
        loss   = bce_weight*loss_f + loss_p

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
        optimizer.step()

        tot     += float(loss.item())
        tot_p   += float(loss_p.item())
        tot_f   += float(bce_weight*loss_f.item())
        n_batches += 1

    sums = torch.tensor([tot, tot_p,  tot_f, n_batches],
                        dtype=torch.float32, device=device)
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)

    denom = max(1.0, sums[3].item())
    return {
        "total":   (sums[0].item() / denom),
        "p_loss":  (sums[1].item() / denom),
        "f_loss":  (sums[2].item() / denom),
    }







def validate(
    model, dataloader, device, epoch, trunk_inputs_encoded,
    p_mean, p_std, world_size=1, num_space_points_val=100,
    lumped_probs: torch.Tensor | None = None,      # <- pass probs ∝ lumped masses, shape [n_space], sum==1
    bce_weight: float = 0.1, gamma: float = 1.0,   # match your training blend
):
    """
    Importance-sampled validation:
      - sample 'num_space_points_val' spatial nodes with probs ~ lumped mass
      - use ALL time steps (uniform)
      - keep normalized quantities; pressure terms scaled by σ²
    """
    model.eval()
    n_space, n_time = trunk_inputs_encoded.shape[:2]
    assert lumped_probs is not None and lumped_probs.numel() == n_space, \
        "Provide `lumped_probs` over spatial nodes (sum==1)."

    total_loss = total_p_loss = total_f_loss = 0.0
    n_batches = 0

    trunk_enc_dev = trunk_inputs_encoded.to(device)

    with torch.no_grad():
        for branch_batch, scalar_batch, target_batch in dataloader:
            branch_batch = branch_batch.to(device, non_blocking=True)
            scalar_batch = scalar_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            B = branch_batch.size(0)
            # reshape target: [B, space, time, 2]
            target_batch = target_batch.view(B, n_space, n_time, target_batch.shape[-1])

            # --- importance-sample spatial nodes once per batch (with replacement) ---
            # (same idx for the whole batch keeps variance lower and makes things comparable)
            idx_space = torch.multinomial(lumped_probs, num_space_points_val, replacement=True)  # [num_space_points_val]

            # --- build trunk_sub & target_sub over ALL times for those space nodes ---
            # trunk: [numS, nT, D] -> [nT*numS, D] then tile over batch
            trunk_sub = trunk_enc_dev[idx_space, :, :]                          # [numS, nT, D]
            trunk_sub = trunk_sub.permute(1, 0, 2).reshape(n_time * num_space_points_val, -1)
            trunk_sub = trunk_sub.unsqueeze(0).expand(B, -1, -1).contiguous()   # [B, nT*numS, D]

            # targets: [B, numS, nT, 2] -> [B, nT*numS, 2]
            target_sub = target_batch[:, idx_space, :, :]                       # [B, numS, nT, 2]
            target_sub = target_sub.permute(0, 2, 1, 3).reshape(B, -1, target_batch.shape[-1])

            # forward
            branch_batch = add_positional_channels(branch_batch)
            p_pred, f_pred = model(branch_batch, scalar_batch, trunk_sub)       # [B, nT*numS, 1] each

            p_true = target_sub[..., 0:1]
            f_true = target_sub[..., 1:2]

            loss_p =loss_pressure(p_pred, p_true)
            f = torch.sigmoid(f_pred)                 # compare probs, not logits
            loss_f = F.mse_loss(f, f_true, reduction='mean')
            loss   = bce_weight*loss_f + loss_p

            total_loss       += float(loss.item())
            total_p_loss     += float(loss_p.item())
            total_f_loss     += float(bce_weight*loss_f.item())
            n_batches += 1

    # Aggregate across ranks if DDP
    dev = next(model.parameters()).device
    local = torch.tensor(
        [total_loss, total_p_loss, total_f_loss,  n_batches],
        dtype=torch.float32, device=dev
    )
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)

    denom = max(1.0, local[3].item())
    return {
        "total":   local[0].item() / denom,
        "p_loss":  local[1].item() / denom,
        "f_loss":  local[2].item() / denom,
    }

