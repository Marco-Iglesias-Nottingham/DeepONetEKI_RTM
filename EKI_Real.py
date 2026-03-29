#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 12:14:36 2025

@author: pmzmi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU-first posterior ensemble (REAL data)
- Mirrors code A's GPU behavior, solvers, and dtype handling
- Uses Real{file_no}.mat and Standard{file_no}.mat
"""

import os
import argparse
import numpy as np
import torch
import h5py
import scipy.io as sio
from scipy.interpolate import interp1d
import time
import random
import pickle
from emulator_tools.models import DeepONetPressureFront
from emulator_tools.data_utils import add_positional_channels

# ---------------------------
# Fast matmul / TF32 like A
# ---------------------------
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------
# Args & path utilities
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name", required=True, help="Base experiment name used in output_<exp-name>")
    p.add_argument("--epoch", type=int, required=True, help="Epoch to load (nn checkpoint & UQ files)")
    p.add_argument("--index-file", type=str, default=None, help="Path to .mat file containing spatial index vector")
    p.add_argument("--index-var", type=str, default="ind", help="Variable name of indices in the .mat file")
    p.add_argument("--prior-file", type=str, default="prior_ensemble.h5", help="Prior ensemble HDF5")
    p.add_argument("--posterior-out", type=str, default=None, help="Optional explicit output path for posterior .h5")

    # NEW: which real data file to use
    p.add_argument("--file-no", type=int, required=True, help="Real/Standard file number (e.g. 1..4)")

    # A-style flags
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--deterministic", action="store_true", help="Deterministic algorithms (slower)")
    p.add_argument("--D", type=int, default=400, help="Shared latent width for model (Dp=Df=D)")

    return p.parse_args()

def choose_tag_from_index(idx_tag):
    """
    Map the --index-file name to the tag used inside the PKL.
    indices_for_real.mat  -> 'for_real'
    indices_100.mat       -> 'idx100'
    """
    if not idx_tag:
        return None
    s = os.path.basename(idx_tag).lower()
    if "for_real" in s:
        return "for_real"
    if "100" in s:
        return "idx100"
    return None


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

def resolve_model_path(model_dir, epoch, idx_tag):
    cands = []
    if idx_tag:
        cands.append(os.path.join(model_dir, f"deeponet_{idx_tag}_epoch_{epoch:03d}.pt"))
    cands.append(os.path.join(model_dir, f"deeponet_epoch_{epoch:03d}.pt"))
    path = resolve_first_existing(cands)
    if path is None:
        raise FileNotFoundError(
            f"Could not find model checkpoint in {model_dir} for epoch={epoch}. Tried:\n" + "\n".join(cands)
        )
    return path

def resolve_uq_path(model_dir, epoch):
    # new location & name used by test_model.py
    return os.path.join(model_dir, "test_outputs_epoch",
                        f"errors_and_samples_epoch_{epoch:03d}.pkl")



def posterior_filename(file_no):
    return f"posterior_real{file_no}.h5"

# ---------------------------
# Torch helpers (A-style)
# ---------------------------
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
        lo_t = torch.as_tensor(lo, dtype=v.dtype, device=v.device)
        hi_t = torch.as_tensor(hi, dtype=v.dtype, device=v.device)
        v[:, n] = torch.log((hi_t - v[:, n]) / (v[:, n] - lo_t))
    return v

def InvTransformAll(variable: torch.Tensor, lim):
    v = variable.clone()
    for n, (lo, hi) in enumerate(lim):
        lo_t = torch.as_tensor(lo, dtype=v.dtype, device=v.device)
        hi_t = torch.as_tensor(hi, dtype=v.dtype, device=v.device)
        e = torch.exp(v[:, n])
        v[:, n] = (hi_t + e * lo_t) / (1.0 + e)
    return v

def create_mask_RT(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def):
    device = LS.device
    dtype  = LS.dtype
    N, H, W = LS.shape
    y_vals = torch.linspace(0, 0.3, H, device=device, dtype=dtype).view(1, H, 1)
    RF_geo_top_exp    = RF_geo_bottom.unsqueeze(1).expand(N, H, W)
    RF_geo_bottom_exp = RF_geo_top.unsqueeze(1).expand(N, H, W)
    mask_top    = y_vals >= (0.3 - RF_geo_top_exp)
    mask_bottom = y_vals <= RF_geo_bottom_exp
    return (mask_top | mask_bottom).to(dtype)


def create_mask(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def):
    device_ = LS.device
    dtype_  = LS.dtype
    N, H, W = LS.shape

    y_vals = torch.linspace(0, 0.3, H, device=device_, dtype=dtype_).view(1, H, 1)

    RF_geo_top_exp    = RF_geo_bottom.unsqueeze(1).expand(N, H, W)
    RF_geo_bottom_exp = RF_geo_top.unsqueeze(1).expand(N, H, W)

    mask_top    = y_vals >= (0.3 - RF_geo_top_exp)
    mask_bottom = y_vals <= RF_geo_bottom_exp
    mask_LS_05  = LS > 1.0

    perm_C     = vec[:, 0].view(N, 1, 1).expand(N, H, W)
    poro_C     = vec[:, 2].view(N, 1, 1).expand(N, H, W)
    poro_def   = vec[:, 3].view(N, 1, 1).expand(N, H, W)
    poro_RT_top    = vec[:, 4].view(N, 1, 1).expand(N, H, W)
    poro_RT_bottom = vec[:, 5].view(N, 1, 1).expand(N, H, W)

    base_perm = torch.where(mask_LS_05, perm_def, perm_C)
    base_poro = torch.where(mask_LS_05, poro_def, poro_C)

    mask_perm = torch.where(mask_top, RT_field_bottom, base_perm)
    mask_perm = torch.where(mask_bottom, RT_field_top, mask_perm)

    mask_poro = torch.where(mask_top, poro_RT_bottom, base_poro)
    mask_poro = torch.where(mask_bottom, poro_RT_top, mask_poro)

    return mask_perm, mask_poro

def set_reproducible(seed: int, deterministic: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

def _sanitize_symmetric(A: torch.Tensor):
    A = 0.5 * (A + A.T)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    s = torch.norm(A, p=float('inf'))
    scale = torch.clamp(s, min=1.0)
    return A / scale, scale

def spd_solve_robust(A, B, rel_floor=1e-6, abs_floor=1e-10, prefer_dtype=torch.float32):
    device_ = A.device
    I = torch.eye(A.shape[0], device=device_, dtype=prefer_dtype)

    A_s, scale = _sanitize_symmetric(A.to(dtype=prefer_dtype))
    B_s = B.to(dtype=prefer_dtype) / scale

    info = {'stage': None, 'lam_min_pre': None, 'lam_floor': None, 'scale': float(scale)}

    A_s = A_s + rel_floor * I

    try:
        w, Q = torch.linalg.eigh(A_s)
        info['stage'] = 'eigh32' if prefer_dtype == torch.float32 else 'eighX'
    except RuntimeError:
        try:
            A64 = A.to(dtype=torch.float64, copy=True)
            A64, scale64 = _sanitize_symmetric(A64)
            I64 = torch.eye(A64.shape[0], device=device_, dtype=torch.float64)
            A64 = A64 + rel_floor * I64
            w, Q = torch.linalg.eigh(A64)
            info['stage'] = 'eigh64'
            B_s = B.to(torch.float64) / scale64
            scale = scale64
        except RuntimeError:
            A64 = A.to(dtype=torch.float64, copy=True)
            A64, scale64 = _sanitize_symmetric(A64)
            U, S, Vh = torch.linalg.svd(A64, full_matrices=False)
            floor = max(abs_floor, rel_floor * float(S.max().clamp(min=1.0)))
            S_clipped = torch.clamp(S, min=floor)
            Y = (U.T @ (B.to(torch.float64) / scale64)) / S_clipped.unsqueeze(-1)
            X = (Vh.T @ Y)
            info.update({'stage': 'svd64', 'lam_min_pre': float(S.min().item()), 'lam_floor': float(floor), 'scale': float(scale64)})
            return X, info

    lam_max = torch.clamp(w.max(), min=1.0)
    lam_floor = torch.clamp(rel_floor * lam_max, min=abs_floor)
    info['lam_min_pre'] = float(w.min().item())
    info['lam_floor'] = float(lam_floor.item())

    w_clipped = torch.clamp(w, min=lam_floor)
    Y = Q.T @ B_s
    Y = Y / w_clipped.unsqueeze(-1)
    X = Q @ Y
    return X, info

@torch.no_grad()
def evaluate_inlet_ensemble_torch(vec_for_emulator,  t):
    """
    vec_for_emulator : [N, 5]  (these are the last 5 physical scalars, z-scored)
    scalar_mean/std  : [5]
    t                : [T] (times in seconds, matching ttf)
    Returns: [N, T] inlet predictions
    """
    # un-normalize to physical space
    phys = vec_for_emulator  # [N,5], fp32
    phys = phys.to(torch.float64)

    # Map: full vec indices 6..10 -> columns 0..4 here:
    #   vec[:,7]  -> phys[:,1]   (A_log)
    #   vec[:,10] -> phys[:,4]   (frac)
    #   vec[:,8]  -> phys[:,2]   (lam)
    #   vec[:,9]  -> phys[:,3]   (beta)
    A_log = phys[:, 1]
    frac  = phys[:, 4]
    lam   = phys[:, 2].clamp_min(1e-12)
    beta  = phys[:, 3]

    A  = torch.exp(A_log)                         # [N]
    T  = t.to(torch.float64).unsqueeze(0)         # [1,T]
    A2 = A.unsqueeze(1)                           # [N,1]
    F2 = frac.unsqueeze(1)
    L2 = lam.unsqueeze(1)
    B2 = beta.unsqueeze(1)

    term  = 1.0 - torch.exp(- (T / L2) ** B2)     # [N,T]
    inlet = F2 * A2 + (A2 - F2 * A2) * term       # [N,T]
    return inlet


@torch.no_grad()
def UpdateKalman_Field(F1, fac, B, Delta_Z_scaled):
    N, H, W = F1.shape
    F1_reshape = F1.reshape(N, -1)
    F1_mean = F1_reshape.mean(dim=0, keepdim=True)
    fac_t = torch.as_tensor(fac, dtype=F1.dtype, device=F1.device)
    Deta_F1 = fac_t * (F1_reshape - F1_mean)
    C_u_z = Deta_F1.T @ Delta_Z_scaled.to(Deta_F1.dtype)
    resultU = (C_u_z @ B.to(Deta_F1.dtype)).T
    F1_reshape = F1_reshape - resultU
    return F1_reshape.reshape(N, H, W)

@torch.no_grad()
def UpdateKalman_vector(RFA, fac, B, Delta_Z_scaled):
    fac_t = torch.as_tensor(fac, dtype=RFA.dtype, device=RFA.device)
    RFA_mean = RFA.mean(dim=0, keepdim=True)
    Deta_RFA = fac_t * (RFA - RFA_mean)
    C_RA_z = Deta_RFA.T @ Delta_Z_scaled.to(Deta_RFA.dtype)
    resultRA = (C_RA_z @ B.to(Deta_RFA.dtype)).T
    return RFA - resultRA

@torch.no_grad()
def batched_forward(model, branch_input_batch, branch_scalars, trunk_coords_encoded, p_mean, p_std, th=0.9, bs=256):
    N = branch_input_batch.shape[0]
    SxT = trunk_coords_encoded.shape[0]
    preds = torch.empty(N, SxT, 1, dtype=torch.float32, device=branch_input_batch.device)
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, N, bs):
            j = min(i+bs, N)
            bb = add_positional_channels(branch_input_batch[i:j].float())
            bt = trunk_coords_encoded.unsqueeze(0).expand(j - i, -1, -1).contiguous()
            pp, fl = model(bb, branch_scalars[i:j], bt)
            fp = torch.sigmoid(fl)
            pm = p_mean.to(dtype=pp.dtype, device=pp.device)
            ps = p_std.to(dtype=pp.dtype,  device=pp.device)
            preds[i:j] = (pp * ps + pm) * (fp > th).to(pp.dtype)
    return preds

# ---------------------------
# Core run
# ---------------------------
def run(args):
    set_reproducible(args.seed, args.deterministic)

    idx_tag = index_tag_from_path(args.index_file)
    file_no = args.file_no

    # Directories
    base_dir = f"output_{args.exp_name}"
    model_dir =  base_dir

    # Normalization stats -> GPU tensors
    stats_path = os.path.join(base_dir, "normalisation_data.pt")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Missing normalization stats: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu")
    branch_min  = stats["branch_min"].to(device=device, dtype=torch.float32)
    branch_max  = stats["branch_max"].to(device=device, dtype=torch.float32)
    scalar_mean = stats["scalar_mean"].to(device=device, dtype=torch.float32)
    scalar_std  = stats["scalar_std"].to(device=device, dtype=torch.float32)
    p_mean      = stats["target_mean"].to(device=device, dtype=torch.float32)
    p_std       = stats["target_std"].to(device=device, dtype=torch.float32)

    # Indices / nodes
    mat_nodes = sio.loadmat('MATLAB_files_for_emulator/Nodes.mat')
    coord_matrix_all = torch.tensor(mat_nodes['Nodes'], dtype=torch.float32, device=device) / 0.3
    if args.index_file:
        index_data = sio.loadmat(args.index_file)
        if args.index_var not in index_data:
            raise KeyError(f"'{args.index_var}' not found in {args.index_file}. Keys: {list(index_data.keys())}")
        ind = torch.tensor(index_data[args.index_var].squeeze(), dtype=torch.long, device=device)
    else:
        ind = torch.arange(coord_matrix_all.shape[1], dtype=torch.long, device=device)

    # Real data & standard
    syn = sio.loadmat(f"RealData/Real{file_no}.mat")
    inlet = sio.loadmat(f"RealData/Inlet{file_no}.mat")
    #standard_file = sio.loadmat(f"Standard{file_no}.mat")

    # UQ
    uq_path = resolve_uq_path(model_dir, args.epoch)
    with open(uq_path, "rb") as f:
        data_UQ = pickle.load(f)
    print(f"Loaded UQ: {uq_path} (keys: {list(data_UQ.keys())})")
    
    tag_key = choose_tag_from_index(args.index_file)
    
    if tag_key == "for_real":
        cov_np  = data_UQ["for_real_sample_cov"]
        mean_np = data_UQ["for_real_mean_error"]
    elif tag_key == "idx100":
        cov_np  = data_UQ["idx100_sample_cov"]
        mean_np = data_UQ["idx100_mean_error"]
    else:
        raise ValueError(
            f"Could not infer PKL tag from --index-file={args.index_file}. "
            f"Expected filename containing 'for_real' or '100'."
        )
    
    # move to torch
    cov_full        = torch.tensor(cov_np,  dtype=torch.float64, device=device)
    mean_error_full = torch.tensor(mean_np, dtype=torch.float64, device=device)    
    

    # Model
    num_frequencies = 6
    trunk_input_dim = 3 * (1 + 2 * num_frequencies)
    D = int(args.D)
    model = DeepONetPressureFront(
        branch_input_channels=4,
        scalar_dim=5,
        trunk_input_dim=trunk_input_dim,
        Dp=D,
        Df=D,
        num_layers=6,
    ).to(device)

    model_path = resolve_model_path(model_dir, args.epoch, idx_tag)
    print(f"Loading model: {model_path}")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Prior ensemble
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
        LS              = f["/Input3"][idx_sorted]
        RF_geo_top      = f["/Input4"][idx_sorted]
        RF_geo_bottom   = f["/Input5"][idx_sorted]
        RT_field_top    = f["/Input6"][idx_sorted]
        RT_field_bottom = f["/Input7"][idx_sorted]
        vec             = f["/Input8"][idx_sorted]
        perm_def        = f["/Input9"][idx_sorted]

    # -> torch on GPU (float64 for fields; we’ll cast to float32 for NN)
    LS = torch.from_numpy(LS.reshape(num_samples, 120, 120).transpose(0, 2, 1)).to(device=device, dtype=torch.float64)
    RT_field_bottom = torch.from_numpy(RT_field_bottom.reshape(num_samples, 120, 120).transpose(0, 2, 1)).to(device, torch.float64)
    RT_field_top    = torch.from_numpy(RT_field_top.reshape(num_samples, 120, 120).transpose(0, 2, 1)).to(device, torch.float64)
    perm_def        = torch.from_numpy(perm_def.reshape(num_samples, 120, 120).transpose(0, 2, 1)).to(device, torch.float64)
    RF_geo_top    = torch.from_numpy(RF_geo_top).to(device, torch.float64)
    RF_geo_bottom = torch.from_numpy(RF_geo_bottom).to(device, torch.float64)
    vec = torch.from_numpy(vec).to(device, torch.float64)

    # Time grids
    true_times = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,19,21,23,25,27,30,35,40,45,50,55,60,65,70,80,90,100,110])
    real_times = syn['ob_times'].flatten()
    real_data = syn['press_full_corr'].T  # [n_sensors, time_points]
    if file_no in (1,3):
        final_time = 80
    elif file_no in (2,4):
        final_time = 60
    else:
        final_time = 80

    true_times_upto_final = true_times[true_times <= final_time]
    interp_func = interp1d(real_times, real_data, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
    real_data_interp = interp_func(true_times_upto_final)

    inlet_data = inlet['inlet'] # [n_sensors, time_points]
    # --- Inlet data: time x 1 -> 1 x time, then interp on axis=1 ---
    
    inlet_data  = inlet_data.T     


    interp_func2 = interp1d(real_times, inlet_data, kind='linear', axis=1, bounds_error=False, fill_value='extrapolate')
    inlet_data_interp = interp_func2(true_times_upto_final)
    inlet_data = torch.tensor(inlet_data_interp, dtype=torch.float64, device=device)

    true_times_torch = torch.tensor(true_times, dtype=torch.float32, device=device)
    ttf = torch.tensor(true_times_upto_final, dtype=torch.float32, device=device)
    syn_data = torch.tensor(real_data_interp, dtype=torch.float64, device=device)

    # Build whitening from UQ (subselect indices matching sensors x chosen times)
    n_sensors = real_data.shape[0]
    n_times_all = true_times_torch.shape[0]
    time_idx = [int(torch.nonzero(true_times_torch == t, as_tuple=False)[0].item()) for t in ttf]
    obs_indices = []
    for s in range(n_sensors):
        for ti in time_idx:
            obs_indices.append(s * n_times_all + ti)
    obs_indices = torch.tensor(obs_indices, dtype=torch.long, device=device)

#    cov_full = torch.tensor(data_UQ["sample_cov"], dtype=torch.float64, device=device)
#    mean_error_full = torch.tensor(data_UQ["sample_mean"], dtype=torch.float64, device=device)
    cov_sub = cov_full.index_select(0, obs_indices).index_select(1, obs_indices)
    mean_error_sub = mean_error_full.index_select(0, obs_indices)

    # Std model (clip like your code)
    #std_ref = torch.tensor(standard_file['standard'], dtype=torch.float64, device=device)
    noisy_data = syn_data.flatten()  # [n_sensors * len(ttf)]
    std2 = torch.maximum(torch.full_like(noisy_data, 100.0), 0.01 * torch.abs(noisy_data))
    ErrCova = torch.diag(std2**2)

    # --- Sizes ---
    n_press_sensors = syn_data.shape[0]          # e.g., 23
    n_inlet_sensors = inlet_data.shape[0]        # usually 1
    T_sel = syn_data.shape[1]                    # len(ttf)
    
    # --- Observation vector: stack pressure then inlet ---
    noisy_data_press  = syn_data.flatten()                                   # [n_press_sensors*T_sel]
    noisy_data_inlet  = inlet_data.flatten()                                 # [n_inlet_sensors*T_sel]
    noisy_data        = torch.cat([noisy_data_press, noisy_data_inlet], 0)   # [M_all]
    
    # --- Model-error covariance: original sub-cov for pressure, zeros for inlet ---
    # (you already computed obs_indices, cov_sub, mean_error_sub for pressures)
    M_press = noisy_data_press.numel()
    M_inlet = noisy_data_inlet.numel()
    M_all   = M_press + M_inlet
    
    cov_aug = torch.zeros(M_all, M_all, dtype=torch.float64, device=device)
    cov_aug[:M_press, :M_press] = cov_sub                                  # keep pressure block
    # inlet block (and cross) remain zero by construction
    
    # --- Model-error mean: pad zeros for inlet entries ---
    mean_error_aug = torch.cat([
        mean_error_sub, torch.zeros(M_inlet, dtype=torch.float64, device=device)
    ], 0)
    
    # --- Measurement error (diagonal) built from the *augmented* noisy_data ---
    std2 = torch.maximum(torch.full_like(noisy_data, 100.0),
                         0.01 * torch.abs(noisy_data))
    ErrCova = torch.diag(std2**2)                                           # [M_all, M_all]
    
    # --- Whitening on augmented covariance ---
    Total_Cova = ErrCova + cov_aug
    evals, evecs = torch.linalg.eigh(Total_Cova)
    evals = evals.clamp_min(1e-12)
    inv_sqrt = evals.rsqrt()
    
    def apply_inv_std(X):
        Y = X @ evecs
        Y = Y * inv_sqrt
        return Y @ evecs.T
    
    noisy_data_updated = noisy_data - mean_error_aug



    # Transform vec for inversion (torch version)
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

    # Trunk encoding for all selected times
    t = (ttf - true_times_torch.min()) / (true_times_torch.max() - true_times_torch.min())
    trunk_inputs = create_trunk_input2(coord_matrix_all, t, 1, ind)
    trunk_coords_encoded = fourier_encode(trunk_inputs.squeeze(0).to(torch.float32), num_frequencies=6)  # [S*T, Dtrunk]

    # EKI loop (single block over all selected times)
    iter = 0
    t_iter = [0.0]
    Misfit_ave = []
    flag = 0
    loop_t0 = time.perf_counter()

    while flag == 0:
        print(f"\n--- Iteration {iter} ---", flush=True)

        # Build branch inputs
        mask_perm, mask_poro = create_mask(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def)
        mask = torch.stack([mask_perm, mask_poro], dim=1).to(torch.float32)
        branch_inputs = normalize_with_minmax_per_channel(mask, branch_min, branch_max).to(torch.float32)
        branch_input_batch = branch_inputs[:num_samples]

        vec_for_emulator = vec[:, 6:].to(torch.float32)
        branch_scalars = (vec_for_emulator - scalar_mean) / (scalar_std + 1e-12)

        # Emulator predictions (mixed precision)
        predictions = batched_forward(
            model, branch_input_batch, branch_scalars, trunk_coords_encoded, p_mean, p_std, th=0.9, bs=512
        )
        pred_subset = predictions.squeeze(-1).to(torch.float64)
        # Inlet predictions (one per ensemble member over selected times)
        inlet_pred = evaluate_inlet_ensemble_torch(
            vec_for_emulator, ttf
        )   # [N, T_sel]

        pred_all = torch.cat([pred_subset, inlet_pred.reshape(inlet_pred.shape[0], -1)], dim=1)  # [N, M_all]
        # Misfit & stats
        error = (pred_all - noisy_data_updated).contiguous()
        weighted_error = apply_inv_std(error)

        M = weighted_error.shape[1]
        mean_weighted_error = weighted_error.mean(dim=0)
        Misfit = torch.norm(mean_weighted_error, p=2) ** 2 / M
        print(Misfit, flush=True)

        Delta_Z = weighted_error - mean_weighted_error
        Delta_Z_scaled = Delta_Z / np.sqrt(num_samples - 1)

        # C and LM tuning in fp32 like A
        Delta_Z_scaled_f = Delta_Z_scaled.to(torch.float32)
        C = Delta_Z_scaled_f.T @ Delta_Z_scaled_f
        C = 0.5 * (C + C.T)
        Z = weighted_error.to(torch.float32).T
        Z_m = weighted_error.mean(dim=0).to(torch.float32)

        Z_row_norm = torch.norm(weighted_error.to(torch.float32), dim=1)
        beta = (Z_row_norm.square().mean() / M).clamp_min(1e-6)

        tuning = 0.65
        I_M = torch.eye(M, dtype=torch.float32, device=device)
        C_beta = C + beta * I_M
        C_tilde_Zm, info = spd_solve_robust(C_beta, Z_m.unsqueeze(1), rel_floor=1e-6, abs_floor=1e-10, prefer_dtype=torch.float32)
        C_tilde_Zm = C_tilde_Zm.squeeze(1)
        TT = tuning * torch.norm(Z_m) - beta * torch.norm(C_tilde_Zm)
        while TT > 0:
            print(f"  ➤ Increasing beta: {beta.item():.4e}", flush=True)
            beta *= 2.0
            C_beta = C + beta * I_M
            C_tilde_Zm, info = spd_solve_robust(C_beta, Z_m.unsqueeze(1), rel_floor=1e-6, abs_floor=1e-10, prefer_dtype=torch.float32)
            C_tilde_Zm = C_tilde_Zm.squeeze(1)
            TT = tuning * torch.norm(Z_m) - beta * torch.norm(C_tilde_Zm)

        alpha = beta
        alpha_0 = alpha
        if t_iter[iter] + 1 / alpha.item() > 1:
            alpha = torch.tensor(1 / (1 - t_iter[iter]), dtype=alpha.dtype, device=alpha.device)

        # Perturb, then solve for B (fp32)
        Zp = weighted_error.to(torch.float32).T
        alpha_sqrt = alpha.sqrt()
        E = torch.randn(M, num_samples, dtype=torch.float32, device=device) * alpha_sqrt
        E -= E.mean(dim=1, keepdim=True)
        Zp = Zp + E
        A_alpha = C + alpha * I_M
        B, infoB = spd_solve_robust(A_alpha, Zp, rel_floor=1e-6, abs_floor=1e-10, prefer_dtype=torch.float32)
        print(f"  ➤ gain solve stage={infoB['stage']} eigmin_pre={infoB['lam_min_pre']:.2e} "
              f"floor={infoB['lam_floor']:.2e} scale={infoB['scale']:.2e}", flush=True)

        fac = np.sqrt(1 / (num_samples - 1))

        # Updates (remain in field dtype, i.e., float64)
        LS               = UpdateKalman_Field(LS,               fac, B, Delta_Z_scaled)
        RT_field_bottom  = UpdateKalman_Field(RT_field_bottom,  fac, B, Delta_Z_scaled)
        RT_field_top     = UpdateKalman_Field(RT_field_top,     fac, B, Delta_Z_scaled)
        RF_geo_bottom    = UpdateKalman_vector(RF_geo_bottom,   fac, B, Delta_Z_scaled)
        RF_geo_top       = UpdateKalman_vector(RF_geo_top,      fac, B, Delta_Z_scaled)
        perm_def         = UpdateKalman_Field(perm_def,         fac, B, Delta_Z_scaled)

        vec_for_inversion = UpdateKalman_vector(vec_for_inversion, fac, B, Delta_Z_scaled)
        vec               = InvTransformAll(vec_for_inversion, Lim)

        Misfit_ave.append(Misfit.item())
        t_iter.append(t_iter[iter] + 1 / alpha.item())
        print(f"Misfit: {Misfit:.4e}, Alpha: {alpha:.4e}, Time: {t_iter[-1]:.4f}", flush=True)

        if abs(t_iter[-1] - 1) < 1e-8:
            print("✅ Converged!!!")
            flag = 1

        sio.savemat('converged.mat', {
            'Misfit_ave': Misfit_ave,
            't': t_iter,
            'iter': iter,
            'alpha': alpha.item(),
            'alpha_0': alpha_0.item()
        })

        iter += 1

    loop_elapsed = time.perf_counter() - loop_t0
    print(f"[EKI while-loop] total time: {loop_elapsed:.3f} s", flush=True)

    # Final forward pass for predictions to save
    mask_perm, mask_poro = create_mask(vec, LS, RT_field_top, RT_field_bottom, RF_geo_bottom, RF_geo_top, perm_def)
    mask_RT= create_mask_RT(vec,LS,RT_field_top,RT_field_bottom,RF_geo_bottom,RF_geo_top,perm_def)

    mask = torch.stack([mask_perm, mask_poro], dim=1).to(torch.float32)
    branch_inputs = normalize_with_minmax_per_channel(mask, branch_min, branch_max).to(torch.float32)
    branch_input_batch = branch_inputs[:num_samples]
    vec_for_emulator = vec[:, 6:].to(torch.float32)
    branch_scalars = (vec_for_emulator - scalar_mean) / (scalar_std + 1e-12)
    predictions = batched_forward(
        model, branch_input_batch, branch_scalars, trunk_coords_encoded, p_mean, p_std, th=0.9, bs=512
    )
    pred_subset = predictions.squeeze(-1).to(torch.float64)
    inlet_pred = evaluate_inlet_ensemble_torch(vec_for_emulator, ttf)
    pred_all = torch.cat([pred_subset, inlet_pred.reshape(inlet_pred.shape[0], -1)], dim=1)

    with torch.no_grad():
        N, M = pred_all.shape
    
        # Ensure strictly non-negative eigenvalues and match dtypes
        sqrt_evals = evals.clamp_min(1e-12).sqrt().to(dtype=pred_subset.dtype)
        Q = evecs.to(dtype=pred_subset.dtype)
    
        # z ~ N(0, I) row-wise; shape [N, M]
        z = torch.randn(N, M, device=pred_subset.device, dtype=pred_subset.dtype)
    
        # noise has Cov = Q Λ Q^T (i.e., Total_Cova)
        # (z * sqrt_evals) scales columns by sqrt eigenvalues, then rotate by Q^T
        noise = (z * sqrt_evals) @ Q.T
    
        # add zero-mean correlated noise to predictions
        pred_noisy = pred_all + noise
    
    # === Save back to the file you already use for predictions ===
    # Assumes you already have `pred_path` defined earlier in your code.
    # Overwrite (or create) the "Predictions" dataset.
    pred_np = pred_noisy.detach().cpu().numpy()


    # ---------------------------------------------------------------
    # Save outputs (posterior, scalars, predictions) like A
    # ---------------------------------------------------------------
    if args.posterior_out:
        out_path = args.posterior_out
    else:
        out_name = posterior_filename(file_no)
        out_path = os.path.join(model_dir, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    root, ext = os.path.splitext(out_path)
    pred_path = root + "_pred" + ext
    scalars_path = root + "_scalar" + ext
    summary_path= root + "_summary" + ext

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('/Input3', data=LS.detach().cpu().numpy())
        f.create_dataset('/Input4', data=RF_geo_top.detach().cpu().numpy())
        f.create_dataset('/Input5', data=RF_geo_bottom.detach().cpu().numpy())
        f.create_dataset('/Input6', data=RT_field_top.detach().cpu().numpy())
        f.create_dataset('/Input7', data=RT_field_bottom.detach().cpu().numpy())
        f.create_dataset('/Input8', data=vec.detach().cpu().numpy())
        f.create_dataset('/Input9', data=perm_def.detach().cpu().numpy())

    with h5py.File(scalars_path, 'w') as f:
        f.create_dataset('/Input8', data=vec.detach().cpu().numpy())

    with h5py.File(pred_path, 'w') as f:
        f.create_dataset('Predictions', data=pred_np)   # augmented
        # stack real_data (pressures) and inlet_data, flattened the same way
        real_aug = torch.cat([syn_data.flatten(), inlet_data.flatten()], 0)
        f.create_dataset('real_data', data=real_aug.detach().cpu().numpy())
        f.create_dataset('real_time', data=true_times_upto_final)  # same t


#    with h5py.File(pred_path, 'w') as f:
#        f.create_dataset('Predictions', data=pred_subset.detach().cpu().numpy())
#        f.create_dataset('real_data', data=syn_data.detach().cpu().numpy())
#        f.create_dataset('real_time', data=true_times_upto_final)

    print(f"✅ Saved posterior ensemble to: {os.path.abspath(out_path)}")
    print(f"✅ Saved scalars to:          {os.path.abspath(scalars_path)}")
    print(f"✅ Saved predictions to:      {os.path.abspath(pred_path)}")

    mp = mask_perm.detach().cpu().numpy()
    mo = mask_poro.detach().cpu().numpy()
    mrt = mask_RT.detach().cpu().numpy()
    perm_std  = np.std(mp, axis=0)
    perm_mean = np.mean(mp, axis=0)
    poro_std  = np.std(mo, axis=0)
    poro_mean = np.mean(mo, axis=0)
    def compute_indicator_mean_np(data_3d_np):
        return (data_3d_np > 1).mean(axis=0)
    mean_LS = compute_indicator_mean_np(LS.detach().cpu().numpy())
    mean_RT = np.mean(mrt, axis=0)    
    # ---- summary arrays ----

       # Save summary to base_dir with file_no reflected
    summary_name = f"summary_vars_real{args.file_no}.npz"
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

    print(f"✅ Saved summary vars to: {os.path.abspath(summary_path)}")
  
    print(f"✅ Saved summary vars to: {os.path.abspath(summary_path)}")
# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    import sys
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision('high')

    if len(sys.argv) > 1:
        args = parse_args()
        run(args)
    else:
        class A:
            exp_name     = "Final2"
            epoch        = 600
            finetuned    = False
            index_file   = "indices_for_real.mat"
            index_var    = "ind"
            file_no      = 3
            prior_file   = "prior_ensemble_5000.h5"
            posterior_out = None
            seed = 42
            deterministic = False
            D = 400
        run(A)
