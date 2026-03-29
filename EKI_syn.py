#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Posterior ensemble generation with CLI options for epoch, finetune, and index file.

Uses:
  - Normalisation stats from:   output_{exp_name}/normalisation_data.pt
  - UQ (covariance/mean) from:  output_{exp_name}/test_outputs_epoch/errors_and_samples_epoch_XXX.pkl
  - Trained model from:         output_{exp_name}[/_finetune]/deeponet_*.pt

The covariance used is always the *filtered* one produced by test_model.py:
  - for_real_sample_cov / for_real_mean_error
  - idx100_sample_cov / idx100_mean_error
"""

import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn.functional as F  # noqa
from torch.utils.data import DataLoader, TensorDataset  # noqa

import h5py
import scipy.io as sio
import pickle


from emulator_tools.models import DeepONetPressureFront
from emulator_tools.data_utils import add_positional_channels

# --------------------------------------------------
# Torch / device setup
# --------------------------------------------------

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# Small linear algebra helpers
# --------------------------------------------------

def _sanitize_symmetric(A: torch.Tensor):
    """Symmetrize, zero NaNs/Infs, and return (A_sym, scale) where scale>=1."""
    A = 0.5 * (A + A.T)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.norm(A, p=float("inf"))
    scale = torch.clamp(s, min=1.0)
    return A / scale, scale


def spd_solve_robust(A, B, rel_floor=1e-6, abs_floor=1e-10, prefer_dtype=torch.float32):
    """
    Solve A X = B for SPD-ish A using robust path:
      1) sanitize (symmetrize, zero NaN/Inf, scale)
      2) try eigh in preferred dtype
      3) if needed, retry in float64
      4) if still failing, do SVD in float64

    Returns:
      X, info_dict
    """
    device = A.device
    I = torch.eye(A.shape[0], device=device, dtype=prefer_dtype)

    # sanitize & scale
    A_s, scale = _sanitize_symmetric(A.to(dtype=prefer_dtype))
    B_s = B.to(dtype=prefer_dtype) / scale

    info = {"stage": None, "lam_min_pre": None, "lam_floor": None, "scale": float(scale)}

    # add a tiny diagonal before eig
    A_s = A_s + rel_floor * I

    # === try eigh in fp32 ===
    try:
        w, Q = torch.linalg.eigh(A_s)
        info["stage"] = "eigh32"
    except RuntimeError:
        # === retry in fp64 ===
        try:
            A64 = A.to(dtype=torch.float64, copy=True)
            A64, scale64 = _sanitize_symmetric(A64)
            I64 = torch.eye(A64.shape[0], device=device, dtype=torch.float64)
            A64 = A64 + rel_floor * I64
            w, Q = torch.linalg.eigh(A64)
            info["stage"] = "eigh64"
            B_s = B.to(torch.float64) / scale64
            scale = scale64
        except RuntimeError:
            # === final fallback: SVD in fp64 ===
            A64 = A.to(dtype=torch.float64, copy=True)
            A64, scale64 = _sanitize_symmetric(A64)
            U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
            floor = max(abs_floor, rel_floor * float(S.max().clamp(min=1.0)))
            S_clipped = torch.clamp(S, min=floor)
            Y = (U.T @ (B.to(torch.float64) / scale64)) / S_clipped.unsqueeze(-1)
            X = (Vh.T @ Y)
            info.update(
                {
                    "stage": "svd64",
                    "lam_min_pre": float(S.min().item()),
                    "lam_floor": float(floor),
                    "scale": float(scale64),
                }
            )
            return X, info

    # eigenvalue flooring path
    lam_max = torch.clamp(w.max(), min=1.0)
    lam_floor = torch.clamp(rel_floor * lam_max, min=abs_floor)
    info["lam_min_pre"] = float(w.min().item())
    info["lam_floor"] = float(lam_floor.item())

    w_clipped = torch.clamp(w, min=lam_floor)
    Y = Q.T @ B_s
    Y = Y / w_clipped.unsqueeze(-1)
    X = Q @ Y
    return X, info


def chol_solve_spd(A, RHS, base_eps=1e-10, max_tries=8):
    """
    Robustly solve (A) X = RHS for SPD-ish A using Cholesky + adaptive jitter.
    A: [M,M], RHS: [M] or [M,N]
    """
    A = A.double()
    RHS = RHS.double()
    A = 0.5 * (A + A.T)
    I = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)

    diag_scale = A.diagonal().abs().mean().clamp_min(1.0)
    eps = (torch.as_tensor(base_eps, dtype=A.dtype, device=A.device) * diag_scale)

    last_err = None
    for _ in range(max_tries):
        try:
            L = torch.linalg.cholesky(A + eps * I)
            X = torch.cholesky_solve(RHS, L)
            return X, eps.item() if isinstance(eps, torch.Tensor) else eps
        except RuntimeError as e:
            last_err = e
            eps = eps * 10.0
    raise RuntimeError(f"Cholesky failed even after jittering. Last error: {last_err}")


# --------------------------------------------------
# Scalar transforms (same as your previous code)
# --------------------------------------------------

def logTrans(variable, lim):
    return np.log((lim[1] - variable) / (variable - lim[0]))


def inverse_log_transform(y, lim):
    e_y = np.exp(y)
    x = (lim[1] + e_y * lim[0]) / (1 + e_y)
    return x


# --------------------------------------------------
# CLI / path helpers
# --------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--exp-name", required=True, help="Base experiment name used in output_<exp-name>"
    )
    p.add_argument(
        "--epoch", type=int, required=True, help="Epoch to load (nn checkpoint & UQ files)"
    )
    p.add_argument(
        "--finetuned",
        action="store_true",
        help="Use output_<exp-name>_finetune and *_fine_* filenames",
    )
    p.add_argument(
        "--index-file",
        type=str,
        default=None,
        help="Path to .mat file containing spatial index vector",
    )
    p.add_argument(
        "--index-var", type=str, default="ind", help="Variable name of indices in the .mat file"
    )
    p.add_argument(
        "--syn-file",
        type=str,
        default="syndata_for_real.mat",
        help="Synthetic data (syndata) .mat file",
    )
    p.add_argument(
        "--prior-file", type=str, default="prior_ensemble.h5", help="Prior ensemble HDF5"
    )
    p.add_argument(
        "--posterior-out",
        type=str,
        default=None,
        help="Optional explicit output path for posterior .h5",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Enforce deterministic algorithms (slower; may disable some kernels)",
    )
    p.add_argument(
        "--D",
        type=int,
        default=400,
        help="Shared latent width (sets Dp=Df=D). Must match the trained checkpoint.",
    )
    return p.parse_args()


def index_tag_from_path(path):
    if not path:
        return None
    base = os.path.basename(path)
    tag = os.path.splitext(base)[0]
    return tag or None


def resolve_first_existing(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def resolve_model_path(model_dir, epoch, finetuned, idx_tag):
    """
    Look for model checkpoint with or without tag / finetune.
    Matches how main.py saves:
        deeponet_epoch_XXX.pt
    and possibly fine-tuned variants.
    """
    cands = []
    if finetuned:
        if idx_tag:
            cands.append(os.path.join(model_dir, f"deeponet_{idx_tag}_epoch_fine_{epoch:03d}.pt"))
        cands.append(os.path.join(model_dir, f"deeponet_epoch_fine_{epoch:03d}.pt"))
        cands.append(os.path.join(model_dir, "deeponet_epoch_fine.pt"))  # legacy
    else:
        if idx_tag:
            cands.append(os.path.join(model_dir, f"deeponet_{idx_tag}_epoch_{epoch:03d}.pt"))
        cands.append(os.path.join(model_dir, f"deeponet_epoch_{epoch:03d}.pt"))

    path = resolve_first_existing(cands)
    if path is None:
        raise FileNotFoundError(
            f"Could not find model checkpoint in {model_dir} for epoch={epoch}. Tried:\n"
            + "\n".join(cands)
        )
    return path


def resolve_uq_path(model_dir, epoch):
    """UQ files as saved by test_model.py."""
    return os.path.join(
        model_dir,
        "test_outputs_epoch",
        f"errors_and_samples_epoch_{epoch:03d}.pkl",
    )


def choose_tag_from_index(idx_path):
    """
    Map the --index-file name to the tag used inside the PKL (test_model.py).
        indices_for_real.mat  -> 'for_real'
        indices_100.mat       -> 'idx100'
    """
    if not idx_path:
        return None
    s = os.path.basename(idx_path).lower()
    if "for_real" in s:
        return "for_real"
    if "100" in s:
        return "idx100"
    return None


def posterior_filename(idx_tag, finetuned, epoch):
    tag = f"_{idx_tag}" if idx_tag else ""
    fine = "_fine" if finetuned else ""
    return f"posterior_ensemble{tag}{fine}_epoch_{epoch:03d}.h5"


# --------------------------------------------------
# Normalisation helpers
# --------------------------------------------------

def normalize_with_minmax_per_channel(X, X_min, X_max):
    eps = 1e-8
    X_norm = torch.empty_like(X)
    for c in range(X.shape[1]):
        X_norm[:, c, :, :] = (X[:, c, :, :] - X_min[c]) / (X_max[c] - X_min[c] + eps)
    return X_norm


def create_trunk_input2(coord_matrix, t, N, ind):
    x = coord_matrix[0, ind]
    y = coord_matrix[1, ind]
    S = x.shape[0]
    T = t.shape[0]

    x_repeat = x.repeat_interleave(T)
    y_repeat = y.repeat_interleave(T)
    t_tile = t.repeat(S)

    coords = torch.stack([x_repeat, y_repeat, t_tile], dim=1)  # [S*T, 3]
    trunk_input = coords.unsqueeze(0).repeat(N, 1, 1)          # [N, S*T, 3]
    return trunk_input


def fourier_encode(x, num_frequencies=6, include_input=True):
    freqs = 2 ** torch.arange(num_frequencies, device=x.device).float()
    x_proj = x.unsqueeze(-1) * freqs       # [..., D, F]
    x_sin = torch.sin(2 * np.pi * x_proj)  # [..., D, F]
    x_cos = torch.cos(2 * np.pi * x_proj)  # [..., D, F]
    x_fourier = torch.cat([x_sin, x_cos], dim=-1)  # [..., D, 2F]
    x_fourier = x_fourier.view(*x.shape[:-1], -1)
    if include_input:
        x_fourier = torch.cat([x, x_fourier], dim=-1)
    return x_fourier


def TransformAll(variable: torch.Tensor, lim):
    v = variable.clone()
    for n, (lo, hi) in enumerate(lim):
        v[:, n] = torch.log((hi - v[:, n]) / (v[:, n] - lo))
    return v


def InvTransformAll(variable: torch.Tensor, lim):
    v = variable.clone()
    for n, (lo, hi) in enumerate(lim):
        e = torch.exp(v[:, n])
        v[:, n] = (hi + e * lo) / (1.0 + e)
    return v


def create_mask(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def):
    # All inputs must be torch tensors on the SAME device, float32/64
    device = LS.device
    dtype = LS.dtype
    N, H, W = LS.shape

    y_vals = torch.linspace(0, 0.3, H, device=device, dtype=dtype).view(1, H, 1)

    RF_geo_top_exp = RF_geo_bottom.unsqueeze(1).expand(N, H, W)
    RF_geo_bottom_exp = RF_geo_top.unsqueeze(1).expand(N, H, W)

    mask_top = y_vals >= (0.3 - RF_geo_top_exp)
    mask_bottom = y_vals <= RF_geo_bottom_exp
    mask_LS_05 = LS > 1.0

    perm_C = vec[:, 0].view(N, 1, 1).expand(N, H, W)
    poro_C = vec[:, 2].view(N, 1, 1).expand(N, H, W)
    poro_def = vec[:, 3].view(N, 1, 1).expand(N, H, W)
    poro_RT_top = vec[:, 4].view(N, 1, 1).expand(N, H, W)
    poro_RT_bottom = vec[:, 5].view(N, 1, 1).expand(N, H, W)

    base_perm = torch.where(mask_LS_05, perm_def, perm_C)
    base_poro = torch.where(mask_LS_05, poro_def, poro_C)

    mask_perm = torch.where(mask_top, RT_field_bottom, base_perm)
    mask_perm = torch.where(mask_bottom, RT_field_top, mask_perm)

    mask_poro = torch.where(mask_top, poro_RT_bottom, base_poro)
    mask_poro = torch.where(mask_bottom, poro_RT_top, mask_poro)

    return mask_perm, mask_poro


def create_mask_RT(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def):
    device = LS.device
    dtype = LS.dtype
    N, H, W = LS.shape
    y_vals = torch.linspace(0, 0.3, H, device=device, dtype=dtype).view(1, H, 1)

    RF_geo_top_exp = RF_geo_bottom.unsqueeze(1).expand(N, H, W)
    RF_geo_bottom_exp = RF_geo_top.unsqueeze(1).expand(N, H, W)

    mask_top = y_vals >= (0.3 - RF_geo_top_exp)
    mask_bottom = y_vals <= RF_geo_bottom_exp
    return (mask_top | mask_bottom).to(dtype)


@torch.no_grad()
def UpdateKalman_Field(F1, fac, B, Delta_Z_scaled):
    N, H, W = F1.shape
    F1_reshape = F1.reshape(N, -1)
    F1_mean = F1_reshape.mean(dim=0, keepdim=True)

    dtype = F1_reshape.dtype
    device = F1_reshape.device
    fac_t = torch.as_tensor(fac, dtype=dtype, device=device)

    Deta_F1 = fac_t * (F1_reshape - F1_mean)  # [N, H*W]
    C_u_z = Deta_F1.T @ Delta_Z_scaled.to(dtype)  # [H*W, M]
    resultU = (C_u_z @ B.to(dtype)).T  # [N, H*W]

    F1_reshape = F1_reshape - resultU
    return F1_reshape.reshape(N, H, W)


@torch.no_grad()
def UpdateKalman_vector(RFA, fac, B, Delta_Z_scaled):
    dtype = RFA.dtype
    device = RFA.device
    fac_t = torch.as_tensor(fac, dtype=dtype, device=device)

    RFA_mean = RFA.mean(dim=0, keepdim=True)
    Deta_RFA = fac_t * (RFA - RFA_mean)

    C_RA_z = Deta_RFA.T @ Delta_Z_scaled.to(dtype)  # [D, M]
    resultRA = (C_RA_z @ B.to(dtype)).T  # [N, D]

    return RFA - resultRA


def batched_forward(
    model,
    branch_input_batch,
    branch_scalars,
    trunk_coords_encoded,
    p_mean,
    p_std,
    th=0.9,
    bs=256,
):
    N = branch_input_batch.shape[0]
    SxT = trunk_coords_encoded.shape[0]
    preds = torch.empty(N, SxT, 1, dtype=torch.float32, device=branch_input_batch.device)

    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, N, bs):
            j = min(i + bs, N)
            bb = add_positional_channels(branch_input_batch[i:j].float())
            bt = trunk_coords_encoded.unsqueeze(0).expand(j - i, -1, -1).contiguous()
            pp, fl = model(bb, branch_scalars[i:j], bt)
            fp = torch.sigmoid(fl)

            pm = p_mean.to(dtype=pp.dtype, device=pp.device)
            ps = p_std.to(dtype=pp.dtype, device=pp.device)
            preds[i:j] = (pp * ps + pm) * (fp > th).to(pp.dtype)

    return preds


def set_reproducible(seed: int, deterministic: bool = False):
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN & algorithm determinism
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


# --------------------------------------------------
# Core run
# --------------------------------------------------

def run(args):
    set_reproducible(args.seed, args.deterministic)
    idx_tag = index_tag_from_path(args.index_file)

    # Directories
    base_dir = f"output_{args.exp_name}"
    model_dir = f"{base_dir}_finetune" if args.finetuned else base_dir

    # Load normalization stats
    stats_path = os.path.join(base_dir, "normalisation_data.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")

    branch_min = stats["branch_min"].to(device=device, dtype=torch.float32)
    branch_max = stats["branch_max"].to(device=device, dtype=torch.float32)
    scalar_mean = stats["scalar_mean"].to(device=device, dtype=torch.float32)
    scalar_std = stats["scalar_std"].to(device=device, dtype=torch.float32)
    p_mean = stats["target_mean"].to(device=device, dtype=torch.float32)
    p_std = stats["target_std"].to(device=device, dtype=torch.float32)

    # Load nodes & indices
    mat_nodes = sio.loadmat("MATLAB_files_for_emulator/Nodes.mat")
    coord_matrix_all = torch.tensor(
        mat_nodes["Nodes"], dtype=torch.float32, device=device
    ) / 0.3

    if args.index_file:
        index_data = sio.loadmat(args.index_file)
        if args.index_var not in index_data:
            raise KeyError(
                f"'{args.index_var}' not found in {args.index_file}. "
                f"Keys: {list(index_data.keys())}"
            )
        ind = torch.tensor(
            index_data[args.index_var].squeeze(), dtype=torch.long, device=device
        )
    else:
        ind = torch.arange(coord_matrix_all.shape[1], dtype=torch.long, device=device)

    # Load syndata
    syn = sio.loadmat(args.syn_file)

    # Load UQ (uncertainty) data: filtered covariance from test_model.py
    uq_path = resolve_uq_path(model_dir, args.epoch)
    with open(uq_path, "rb") as f:
        data_UQ = pickle.load(f)
    print(f"Loaded UQ: {uq_path} (keys: {list(data_UQ.keys())})")

    tag_key = choose_tag_from_index(args.index_file)
    if tag_key == "for_real":
        cov_np = data_UQ["for_real_sample_cov"]
        mean_np = data_UQ["for_real_mean_error"]
    elif tag_key == "idx100":
        cov_np = data_UQ["idx100_sample_cov"]
        mean_np = data_UQ["idx100_mean_error"]
    else:
        raise ValueError(
            f"Could not infer PKL tag from --index-file={args.index_file}. "
            f"Expected filename containing 'for_real' or '100'."
        )

    cov = torch.tensor(cov_np, dtype=torch.float64, device=device)
    mean_error = torch.tensor(mean_np, dtype=torch.float64, device=device)

    # Load model
    model_path = resolve_model_path(model_dir, args.epoch, args.finetuned, idx_tag)
    print(f"Loading model: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")

    num_frequencies = 6
    trunk_input_dim = 3 * (1 + 2 * num_frequencies)
    D = int(args.D)
    Dp = D
    Df = D

    model = DeepONetPressureFront(
        branch_input_channels=4,
        scalar_dim=5,
        trunk_input_dim=trunk_input_dim,
        Dp=Dp,
        Df=Df,
        num_layers=6,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # ----------- Load prior ensemble -----------
    prior_file = args.prior_file
    if not os.path.exists(prior_file):
        raise FileNotFoundError(f"Missing prior ensemble: {prior_file}")

    num_samples = 5000

    with h5py.File(prior_file, "r") as f:
        N = f["/Input3"].shape[0]
        num_samples = min(num_samples, N)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(N, size=num_samples, replace=False)

        perm = np.argsort(idx)
        idx_sorted = idx[perm]

        LS = f["/Input3"][idx_sorted]
        RF_geo_top = f["/Input4"][idx_sorted]
        RF_geo_bottom = f["/Input5"][idx_sorted]
        RT_field_top = f["/Input6"][idx_sorted]
        RT_field_bottom = f["/Input7"][idx_sorted]
        vec = f["/Input8"][idx_sorted]
        perm_def = f["/Input9"][idx_sorted]

    # reshape then move to GPU as float64
    LS = torch.from_numpy(LS.reshape(num_samples, 120, 120).transpose(0, 2, 1)).to(
        device=device, dtype=torch.float64
    )
    RT_field_bottom = torch.from_numpy(
        RT_field_bottom.reshape(num_samples, 120, 120).transpose(0, 2, 1)
    ).to(device, torch.float64)
    RT_field_top = torch.from_numpy(
        RT_field_top.reshape(num_samples, 120, 120).transpose(0, 2, 1)
    ).to(device, torch.float64)
    perm_def = torch.from_numpy(
        perm_def.reshape(num_samples, 120, 120).transpose(0, 2, 1)
    ).to(device, torch.float64)

    RF_geo_top = torch.from_numpy(RF_geo_top).to(device, torch.float64)
    RF_geo_bottom = torch.from_numpy(RF_geo_bottom).to(device, torch.float64)
    vec = torch.from_numpy(vec).to(device, torch.float64)

    # ---------- time setup ----------
    true_times = np.array(
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            19,
            21,
            23,
            25,
            27,
            30,
            35,
            40,
            45,
            50,
            55,
            60,
            65,
            70,
            80,
            90,
            100,
            110,
        ]
    )
    true_times = torch.tensor(true_times, dtype=torch.float32, device=device)
    t = (true_times - true_times.min()) / (true_times.max() - true_times.min())

    # ---------- transform vec ----------
    Lim = [
        [np.log(2.5e-10), np.log(6.5e-10)],
        [np.log(0.25e-10), np.log(2.5e-10)],
        [0.6, 0.8],
        [0.55, 0.7],
        [0.9, 0.96],
        [0.9, 0.96],
        [0.085, 0.12],
        [np.log(92e3), np.log(120e3)],
        [0.6, 1.25],
        [0.2, 0.7],
        [0.35, 0.75],
    ]
    vec_for_inversion = TransformAll(vec, Lim)

    # Masks for network branch inputs (perm & poro) for quick probe
    mask1 = torch.log(
        torch.as_tensor(syn["mask"], dtype=torch.float32, device=device)
    ).unsqueeze(0)
    mask2 = torch.as_tensor(syn["mask2"], dtype=torch.float32, device=device).unsqueeze(
        0
    )
    mask_probe = torch.stack([mask1, mask2], dim=1)  # [1, 2, H, W]

    branch_inputs_probe = normalize_with_minmax_per_channel(
        mask_probe.to(torch.float64), branch_min, branch_max
    )

    trunk_probe = create_trunk_input2(coord_matrix_all, t, 1, ind)
    trunk_inputs_encoded = fourier_encode(
        trunk_probe.squeeze(0).to(torch.float32), num_frequencies=6
    )

    scalars_probe = torch.tensor(
        [0.0922, np.log(1.0912e5), 1.114, 0.42, 0.66],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    branch_scalars_probe = (
        (scalars_probe.double() - scalar_mean) / (scalar_std + 1e-12)
    ).float()

    # Quick probe
    with torch.no_grad():
        bb = add_positional_channels(branch_inputs_probe[:1].float())
        tt = trunk_inputs_encoded.unsqueeze(0)
        ss = branch_scalars_probe[:1]
        p_pred, f_logit = model(bb, ss, tt)
        f_pred = torch.sigmoid(f_logit)

        pm = p_mean.to(dtype=p_pred.dtype, device=p_pred.device).view(1, 1, 1)
        ps = p_std.to(dtype=p_pred.dtype, device=p_pred.device).view(1, 1, 1)
        mask_front = (f_pred > 0.5).to(dtype=p_pred.dtype)
        p_pred_un = (p_pred * ps + pm) * mask_front
        print("Probe prediction shape:", p_pred_un.shape, flush=True)

    # -------- UQ / data for inversion ----------
    syn_data = torch.tensor(
        syn["syn_data"], dtype=torch.float64, device=device
    )  # [M]
    noise_free_data = torch.tensor(
        syn["noise_free_data"], dtype=torch.float64, device=device
    )  # noqa: F841
    std = torch.tensor(syn["Error_std"], dtype=torch.float64, device=device)

    n_times = 34
    n_space = 2973

    batch_size = 100
    time_step = n_times - 1

    # Observation covariance & whitening
    ErrCova = torch.diag(std.flatten() ** 2)
    noisy_data = syn_data.flatten()
    M = noisy_data.shape[0]

    Total_Cova = ErrCova + cov
    evals, evecs = torch.linalg.eigh(Total_Cova)
    evals = evals.clamp_min(1e-12)
    inv_sqrt = evals.rsqrt()

    def apply_inv_std(X):
        Y = X @ evecs
        Y = Y * inv_sqrt
        return Y @ evecs.T

    noisy_data_updated = noisy_data - mean_error

    iter = 0
    t_iter = [0.0]
    Misfit_ave = []
    flag = 0

    trunk_inputs_1 = create_trunk_input2(coord_matrix_all, t[0 : time_step + 1], 1, ind)
    trunk_coords_encoded = fourier_encode(
        trunk_inputs_1.squeeze(0).to(torch.float32).to(device), num_frequencies=6
    )

    loop_t0 = time.perf_counter()

    while flag == 0:
        print(f"\n--- Iteration {iter}")

        # 1) Build inputs
        mask_perm, mask_poro = create_mask(
            vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def
        )
        mask = torch.stack([mask_perm, mask_poro], dim=1).to(torch.float32)

        branch_inputs = normalize_with_minmax_per_channel(
            mask, branch_min, branch_max
        ).to(torch.float32)
        branch_input_batch = branch_inputs[:num_samples]

        vec_for_emulator = vec[:, 6:].to(torch.float32)
        branch_scalars = (vec_for_emulator - scalar_mean) / (scalar_std + 1e-12)

        predictions = batched_forward(
            model,
            branch_input_batch,
            branch_scalars,
            trunk_coords_encoded,
            p_mean,
            p_std,
            th=0.9,
            bs=512,
        )

        pred_subset = predictions.squeeze(-1).to(noisy_data_updated.dtype)

        # 3) Misfit and EKI stats
        error = (pred_subset - noisy_data_updated).contiguous()
        weighted_error = apply_inv_std(error)

        mean_weighted_error = weighted_error.mean(dim=0)
        Misfit = torch.norm(mean_weighted_error, p=2) ** 2 / M
        print(Misfit, flush=True)

        if Misfit < 1 and t_iter[iter] > 1:
            print("Stopping early: Misfit < 1")
            break

        Delta_Z = weighted_error - mean_weighted_error
        Delta_Z_scaled = Delta_Z / np.sqrt(num_samples - 1)

        Delta_Z_scaled_f = Delta_Z_scaled.to(torch.float32)
        M_dim = Delta_Z_scaled_f.shape[1]

        C = Delta_Z_scaled_f.T @ Delta_Z_scaled_f  # [M,M]
        C = 0.5 * (C + C.T)
        Z = weighted_error.T
        Z_m = weighted_error.mean(dim=0)

        Z_row_norm = torch.norm(weighted_error, dim=1)
        beta = (Z_row_norm.square().mean() / M_dim).clamp_min(1e-6)

        tuning = 0.75
        I_M = torch.eye(M_dim, dtype=torch.float32, device=device)
        C_beta = C + beta * I_M

        C_tilde_Zm, info = spd_solve_robust(
            C_beta,
            Z_m.unsqueeze(1),
            rel_floor=1e-6,
            abs_floor=1e-10,
            prefer_dtype=torch.float32,
        )
        C_tilde_Zm = C_tilde_Zm.squeeze(1)
        print(
            f"  ➤ LM solve stage={info['stage']} eigmin_pre={info['lam_min_pre']:.2e} "
            f"floor={info['lam_floor']:.2e} scale={info['scale']:.2e}",
            flush=True,
        )

        TT = tuning * torch.norm(Z_m) - beta * torch.norm(C_tilde_Zm)
        while TT > 0:
            print(f"  ➤ Increasing beta: {beta.item():.4e}", flush=True)
            beta *= 2.0
            C_beta = C + beta * I_M
            C_tilde_Zm, info = spd_solve_robust(
                C_beta,
                Z_m.unsqueeze(1),
                rel_floor=1e-6,
                abs_floor=1e-10,
                prefer_dtype=torch.float32,
            )
            C_tilde_Zm = C_tilde_Zm.squeeze(1)
            print(
                f"    · stage={info['stage']} eigmin_pre={info['lam_min_pre']:.2e} "
                f"floor={info['lam_floor']:.2e} scale={info['scale']:.2e}",
                flush=True,
            )
            TT = tuning * torch.norm(Z_m) - beta * torch.norm(C_tilde_Zm)

        alpha = beta
        alpha_0 = alpha
        if t_iter[iter] + 1 / alpha.item() > 1:
            alpha = torch.tensor(
                1 / (1 - t_iter[iter]),
                dtype=alpha.dtype,
                device=alpha.device,
            )

        Z = weighted_error.to(torch.float32).T

        alpha_sqrt = alpha.sqrt()
        E = torch.randn_like(Z) * alpha_sqrt
        E -= E.mean(dim=1, keepdim=True)
        Z = Z + E

        A_alpha = C + alpha * I_M
        B, infoB = spd_solve_robust(
            A_alpha,
            Z,
            rel_floor=1e-6,
            abs_floor=1e-10,
            prefer_dtype=torch.float32,
        )
        print(
            f"  ➤ gain solve stage={infoB['stage']} eigmin_pre={infoB['lam_min_pre']:.2e} "
            f"floor={infoB['lam_floor']:.2e} scale={infoB['scale']:.2e}",
            flush=True,
        )

        fac = np.sqrt(1 / (num_samples - 1))

        # Update fields and vectors
        LS = UpdateKalman_Field(LS, fac, B, Delta_Z_scaled)
        RT_field_bottom = UpdateKalman_Field(
            RT_field_bottom, fac, B, Delta_Z_scaled
        )
        RT_field_top = UpdateKalman_Field(
            RT_field_top, fac, B, Delta_Z_scaled
        )
        RF_geo_bottom = UpdateKalman_vector(
            RF_geo_bottom, fac, B, Delta_Z_scaled
        )
        RF_geo_top = UpdateKalman_vector(
            RF_geo_top, fac, B, Delta_Z_scaled
        )
        perm_def = UpdateKalman_Field(
            perm_def, fac, B, Delta_Z_scaled
        )
        vec_for_inversion = UpdateKalman_vector(
            vec_for_inversion, fac, B, Delta_Z_scaled
        )
        vec = InvTransformAll(vec_for_inversion, Lim)

        Misfit_ave.append(Misfit.item())
        t_iter.append(t_iter[iter] + 1 / alpha.item())
        print(
            f"Misfit: {Misfit:.4e}, Alpha: {alpha:.4e}, Time: {t_iter[-1]:.4f}",
            flush=True,
        )

        if abs(t_iter[-1] - 1) < 1e-8:
            print("Converged")
            flag = 1

        sio.savemat(
            "converged.mat",
            {
                "Misfit_ave": Misfit_ave,
                "t": t_iter,
                "iter": iter,
                "alpha": alpha.item(),
                "alpha_0": alpha_0.item(),
            },
        )

        iter += 1

    loop_elapsed = time.perf_counter() - loop_t0
    print(f"[EKI while-loop] total time: {loop_elapsed:.3f} s", flush=True)

    # Final fields to save
    mask_perm, mask_poro = create_mask(
        vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def
    )
    mask_RT = create_mask_RT(
        vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def
    )
    mask_final = torch.stack([mask_perm, mask_poro], dim=1).to(torch.float32)

    branch_inputs = normalize_with_minmax_per_channel(
        mask_final, branch_min, branch_max
    ).to(torch.float32)
    branch_input_batch = branch_inputs[:num_samples]

    vec_for_emulator = vec[:, 6:].to(torch.float32)
    branch_scalars = (vec_for_emulator - scalar_mean) / (scalar_std + 1e-12)

    predictions = batched_forward(
        model,
        branch_input_batch,
        branch_scalars,
        trunk_coords_encoded,
        p_mean,
        p_std,
        th=0.9,
        bs=512,
    )

    pred_subset = predictions.squeeze(-1).to(noisy_data_updated.dtype)

    # ---------------------------------------------------------------
    # Save the posterior ensemble
    # ---------------------------------------------------------------
    if args.posterior_out:
        out_path = args.posterior_out
    else:
        out_name = posterior_filename(idx_tag, args.finetuned, args.epoch)
        out_path = os.path.join(model_dir, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root, ext = os.path.splitext(out_path)
    pred_path = root + "_pred" + ext
    scalars_path = root + "_scalar" + ext

    with h5py.File(out_path, "w") as f:
        f.create_dataset("/Input3", data=LS.detach().cpu().numpy())
        f.create_dataset("/Input4", data=RF_geo_top.detach().cpu().numpy())
        f.create_dataset("/Input5", data=RF_geo_bottom.detach().cpu().numpy())
        f.create_dataset("/Input6", data=RT_field_top.detach().cpu().numpy())
        f.create_dataset("/Input7", data=RT_field_bottom.detach().cpu().numpy())
        f.create_dataset("/Input8", data=vec.detach().cpu().numpy())
        f.create_dataset("/Input9", data=perm_def.detach().cpu().numpy())

    with h5py.File(scalars_path, "w") as f:
        f.create_dataset("/Input8", data=vec.detach().cpu().numpy())

    print(f"Saved posterior ensemble to: {os.path.abspath(out_path)}")

    # summary vars
    mp = mask_perm.detach().cpu().numpy()
    mo = mask_poro.detach().cpu().numpy()
    mrt = mask_RT.detach().cpu().numpy()
    perm_std = np.std(mp, axis=0)
    perm_mean = np.mean(mp, axis=0)
    poro_std = np.std(mo, axis=0)
    poro_mean = np.mean(mo, axis=0)

    def compute_indicator_mean_np(data_3d_np):
        return (data_3d_np > 1).mean(axis=0)

    mean_LS = compute_indicator_mean_np(LS.detach().cpu().numpy())
    mean_RT = np.mean(mrt, axis=0)

    summary_name = posterior_filename(idx_tag, args.finetuned, args.epoch).replace(
        "posterior_ensemble", "summary_vars"
    ).replace(".h5", ".npz")
    summary_path = os.path.join(base_dir, summary_name)

    np.savez(
        summary_path,
        perm_std=perm_std,
        perm_mean=perm_mean,
        poro_std=poro_std,
        poro_mean=poro_mean,
        mean_LS=mean_LS,
        mean_RT=mean_RT,
    )
    print(f"Saved summary vars to: {os.path.abspath(summary_path)}")

    # ---------------------------------------------------
    # Add correlated noise to predictions and save
    # ---------------------------------------------------
    with torch.no_grad():
        N_pred, M_pred = pred_subset.shape

        sqrt_evals = evals.clamp_min(1e-12).sqrt().to(dtype=pred_subset.dtype)
        Q = evecs.to(dtype=pred_subset.dtype)

        z = torch.randn(N_pred, M_pred, device=pred_subset.device, dtype=pred_subset.dtype)
        noise = (z * sqrt_evals) @ Q.T
        pred_noisy = pred_subset + noise

    pred_np = pred_noisy.detach().cpu().numpy()
    # use 'a' so the file is created if it doesn't exist yet
    with h5py.File(pred_path, "a") as f:
        if "Predictions" in f:
            del f["Predictions"]
        f.create_dataset("Predictions", data=pred_np, compression="gzip", compression_opts=4)


# --------------------------------------------------
# Entrypoint
# --------------------------------------------------

if __name__ == "__main__":
    import sys

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")

    if len(sys.argv) > 1:
        args = parse_args()
        run(args)
    else:
        # Quick debug defaults when running from an IDE
        class A:
            exp_name = "NS40000"
            epoch = 400
            finetuned = False
            index_file = "indices_100.mat"
            index_var = "ind"
            syn_file = "MATLAB_files_for_emulator/syndata_100.mat"
            prior_file = "Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5"
            posterior_out = None
            seed = 42
            deterministic = False
            D = 400

        run(A)

# %%
