import torch
import numpy as np
import scipy.io as sio
import h5py
import random
import os
from scipy.io import loadmat



def make_lumped_mass_probs(mat_path: str, device: torch.device) -> torch.Tensor:
    """Return normalized lumped-mass probabilities over spatial nodes [S] (sum==1)."""
    M = loadmat(mat_path)['MassMatrix']                    # scipy sparse or dense
    m = np.array(M.sum(axis=1)).ravel().astype(np.float64) # row-sum lumping
    m = m / m.sum()
    return torch.tensor(m, dtype=torch.float32, device=device)


def create_trunk_input_indices(coord_matrix, t, N,ind):
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



def normalize_and_standardize(
        
    train_branch, val_branch, test_branch,
    train_scalars, val_scalars, test_scalars,
    train_targets, val_targets, test_targets,
    output_stats_file="normalisation_data.pt",    
):
    """
    Normalize branch inputs, standardize scalars & pressure targets.
    Saves normalization stats for inference.

    Returns:
        normalized (branch, scalars, targets) and stats dict.
    """

    # branch min-max normalization
    X_min = train_branch.amin(dim=(0, 2, 3))
    X_max = train_branch.amax(dim=(0, 2, 3))
    eps = 1e-8
    def minmax(x):
        x_norm = torch.empty_like(x)
        for c in range(x.shape[1]):
            x_norm[:, c] = (x[:, c] - X_min[c]) / (X_max[c] - X_min[c] + eps)
        return x_norm

    train_branch_norm = minmax(train_branch)
    val_branch_norm = minmax(val_branch)
    test_branch_norm = minmax(test_branch)

    # scalars standardization
    scalar_mean = train_scalars.mean(dim=0)
    scalar_std = train_scalars.std(dim=0)

    train_scalars_norm = (train_scalars - scalar_mean) / scalar_std
    val_scalars_norm = (val_scalars - scalar_mean) / scalar_std
    test_scalars_norm = (test_scalars - scalar_mean) / scalar_std

    # pressure standardization
    train_p = train_targets[..., 0]
    train_f = train_targets[..., 1]
    val_p = val_targets[..., 0]
    val_f = val_targets[..., 1]
    test_p = test_targets[..., 0]
    test_f = test_targets[..., 1]

    p_mean = train_p.mean()
    p_std = train_p.std()

    train_p_norm = (train_p - p_mean) / p_std
    val_p_norm = (val_p - p_mean) / p_std
    test_p_norm = (test_p - p_mean) / p_std

    train_targets_norm = torch.stack([train_p_norm, train_f], dim=-1)
    val_targets_norm = torch.stack([val_p_norm, val_f], dim=-1)
    test_targets_norm = torch.stack([test_p_norm, test_f], dim=-1)

    # save stats
    stats = {
        "branch_min": X_min,
        "branch_max": X_max,
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
        "target_mean": p_mean,
        "target_std": p_std
    }
    torch.save(stats, output_stats_file)
    print(f"✅ Saved stats → {output_stats_file}")
    return (
        (train_branch_norm, val_branch_norm, test_branch_norm),
        (train_scalars_norm, val_scalars_norm, test_scalars_norm),
        (train_targets_norm, val_targets_norm, test_targets_norm),
        stats
    )


def apply_saved_normalization(
    branch_inputs, scalar_inputs, targets,
    stats
):
    """
    Apply previously saved normalization stats to new (regenerated) data.
    """
    # Unpack stats
    X_min = stats["branch_min"]
    X_max = stats["branch_max"]
    scalar_mean = stats["scalar_mean"]
    scalar_std = stats["scalar_std"]
    p_mean = stats["target_mean"]
    p_std = stats["target_std"]
    eps = 1e-8

    def minmax(x):
        x_norm = torch.empty_like(x)
        for c in range(x.shape[1]):
            x_norm[:, c] = (x[:, c] - X_min[c]) / (X_max[c] - X_min[c] + eps)
        return x_norm

    # Normalize branch inputs
    branch_norm = minmax(branch_inputs)

    # Standardize scalars
    scalars_norm = (scalar_inputs - scalar_mean) / scalar_std

    # Standardize pressure target only (f stays untouched)
    p = targets[..., 0]
    f = targets[..., 1]
    p_norm = (p - p_mean) / p_std
    targets_norm = torch.stack([p_norm, f], dim=-1)

    return branch_norm, scalars_norm, targets_norm


def normalize_with_minmax_per_channel(X, X_min, X_max):
    """
    Min-max normalize each channel independently.

    Args:
        X: Tensor of shape [N, C, H, W]
        X_min, X_max: Tensors of shape [C], one min/max per channel
    Returns:
        Normalized X of same shape
    """
    eps = 1e-8
    X_norm = torch.empty_like(X)
    for c in range(X.shape[1]):
        X_norm[:, c, :, :] = (X[:, c, :, :] - X_min[c]) / (X_max[c] - X_min[c] + eps)
    return X_norm



def create_trunk_input(coord_matrix, t):
    """
    Create the trunk input grid [S*T, 3], independent of number of samples.

    Args:
        coord_matrix: [2, S] tensor of x and y coordinates
        t: [T] tensor of normalized time steps

    Returns:
        trunk_input: [S*T, 3] tensor of (x,y,t) points
    """
    x = coord_matrix[0, :]  # [S]
    y = coord_matrix[1, :]  # [S]
    S = x.shape[0]
    T = t.shape[0]

    x_repeat = x.repeat_interleave(T)      # [S*T]
    y_repeat = y.repeat_interleave(T)      # [S*T]
    t_tile   = t.repeat(S)                 # [S*T]

    trunk_input = torch.stack([x_repeat, y_repeat, t_tile], dim=1)  # [S*T, 3]
    return trunk_input


def add_positional_channels(field_tensor):
    """
    Adds normalized Y and X positional encodings as extra channels.

    Args:
        field_tensor: Tensor of shape [B, 1, H, W]
    Returns:
        Tensor of shape [B, 3, H, W]
    """
    B, C, H, W = field_tensor.shape
    device = field_tensor.device

    # Y position: vertical
    y_coords = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

    # X position: horizontal
    x_coords = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)

    return torch.cat([field_tensor, y_coords, x_coords], dim=1)  # → [B, 3, H, W]


def fourier_encode(x, num_frequencies=6, include_input=True):
    """
    x: tensor [..., D]
    returns: [..., D * (1 + 2 * num_frequencies)] if include_input
    """
    import numpy as np
    freqs = 2 ** torch.arange(num_frequencies, device=x.device).float()
    x_proj = x.unsqueeze(-1) * freqs  # [..., D, F]
    x_sin = torch.sin(2 * np.pi * x_proj)  # [..., D, F]
    x_cos = torch.cos(2 * np.pi * x_proj)  # [..., D, F]
    x_fourier = torch.cat([x_sin, x_cos], dim=-1)  # [..., D, 2F]
    x_fourier = x_fourier.view(*x.shape[:-1], -1)  # flatten last two dims
    if include_input:
        x_fourier = torch.cat([x, x_fourier], dim=-1)
    return x_fourier


def load_normalization_stats(exp_name, device):
    stats_path = os.path.join(f"output_{exp_name}", "normalisation_data.pt")
    print("normalisation path:", stats_path)
    data = torch.load(stats_path, map_location=device)
    return {
        k: torch.tensor(v, dtype=torch.float64, device=device)
        for k, v in data.items()
    }


def generate_data(
    num_samples=None,          # <- now optional / ignored
    num_points=1,
    chunk_size=500,
    num_frequencies=6,
    num_files=4,
    per_file_capacity=10_000,
    strict_capacity=True,
):

    # ----- geometry & time encoding (unchanged) -----
    mat = sio.loadmat("MATLAB_files_for_emulator/Nodes.mat")
    coord_matrix = torch.tensor(mat["Nodes"], dtype=torch.float32) / 0.3

    true_times = torch.tensor(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            17, 19, 21, 23, 25, 27, 30, 35, 40, 45, 50, 55, 60,
            65, 70, 80, 90, 100, 110
        ],
        dtype=torch.float32,
    )
    t_norm = (true_times - true_times.min()) / (true_times.max() - true_times.min())

    n_times = t_norm.shape[0]
    n_space = coord_matrix.shape[1]

    trunk_inputs_raw = create_trunk_input(coord_matrix, t_norm)
    trunk_inputs_encoded = fourier_encode(trunk_inputs_raw, num_frequencies=num_frequencies)
    trunk_input_dim = 3 * (1 + 2 * num_frequencies)
    trunk_inputs_encoded = trunk_inputs_encoded.view(n_space, n_times, trunk_input_dim)

    # ----- load from a single pair of files -----
    in_path = "input_data_batch_seed_1.h5"
    out_path = "output_data_batch_seed_1.h5"

    print(f"Reading from: {in_path} and {out_path}")
    print(f"Processing in chunks of {chunk_size}...")

    branch_chunks, target_chunks, scalar_chunks = [], [], []

    with h5py.File(out_path, "r") as f_out, h5py.File(in_path, "r") as f_in:
        available = int(f_out["/Output1"].shape[0])
        nsamples = available                 # <-- load ALL available samples

        if nsamples <= 0:
            raise ValueError(f"No samples available in {out_path} (Output1 has size {available}).")

        for i in range(0, nsamples, chunk_size):
            end = min(i + chunk_size, nsamples)

            Pred     = f_out["/Output1"][i:end]
            f_target = f_out["/Output2"][i:end]
            Perm     = f_in["/Input1"][i:end]
            Poro     = f_in["/Input2"][i:end]
            vec      = f_in["/Input8"][i:end]

            # choose the right scalar slice here!
            # Example: all columns:
            # scalars = torch.from_numpy(vec).float()
            scalars = torch.from_numpy(vec[:, :5]).float()  # or whatever matches scalar_dim

            branch = np.stack([Perm, Poro], axis=1)
            branch_tensor = torch.from_numpy(branch).float()

            t1 = Pred.reshape(end - i, -1, 1)
            t2 = f_target.reshape(end - i, -1, 1)
            t1_tensor = torch.from_numpy(t1).float()
            t2_tensor = torch.from_numpy(t2).float()
            target = torch.cat([t1_tensor, t2_tensor], dim=2)

            branch_chunks.append(branch_tensor)
            target_chunks.append(target)
            scalar_chunks.append(scalars)

    branch_inputs    = torch.cat(branch_chunks, dim=0) if branch_chunks else torch.empty(0)
    targets_combined = torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0)
    scalars_combined = torch.cat(scalar_chunks, dim=0) if scalar_chunks else torch.empty(0)

    print("Loaded total samples:", branch_inputs.shape[0])
    print("branch:", branch_inputs.shape)
    return branch_inputs, targets_combined, scalars_combined, trunk_inputs_encoded



def generate_data_for_testing(num_samples, num_points=1, chunk_size=500, num_frequencies=6):
    """
    Load test data from a single pair of HDF5 files:

        input_data_batch_seed2.h5
        output_data_batch_seed2.h5

    and build:
        - branch_inputs      : [N, 2, H, W] (Perm, Poro)
        - targets_combined   : [N, S*T, 2]  (pressure, front)
        - scalars_combined   : [N, n_scalar]
        - trunk_inputs_encoded: [S, T, D_trunk]
    """
    import h5py
    import numpy as np
    import scipy.io as sio
    import torch
    import os

    # ----- geometry & time encoding -----
    # Load Nodes from your MATLAB folder (keep consistent with training)
    mat = sio.loadmat(os.path.join("MATLAB_files_for_emulator", "Nodes.mat"))
    coord_matrix = torch.tensor(mat["Nodes"], dtype=torch.float32) / 0.3

    true_times = torch.tensor(
        [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            17, 19, 21, 23, 25, 27, 30, 35, 40, 45, 50, 55, 60,
            65, 70, 80, 90, 100, 110
        ],
        dtype=torch.float32,
    )
    t_norm = (true_times - true_times.min()) / (true_times.max() - true_times.min())

    n_times = t_norm.shape[0]
    n_space = coord_matrix.shape[1]

    trunk_inputs_raw = create_trunk_input(coord_matrix, t_norm)
    trunk_inputs_encoded = fourier_encode(trunk_inputs_raw, num_frequencies=num_frequencies)
    trunk_input_dim = 3 * (1 + 2 * num_frequencies)
    trunk_inputs_encoded = trunk_inputs_encoded.view(n_space, n_times, trunk_input_dim)

    # ----- HDF5 file pair for testing -----
    in_path = "input_data_batch_seed_2.h5"
    out_path = "output_data_batch_seed_2.h5"

    branch_chunks = []
    target_chunks = []
    scalar_chunks = []

    print(f"Reading test data from:\n  {in_path}\n  {out_path}")
    print(f"Processing in chunks of {chunk_size}...")

    with h5py.File(out_path, "r") as f_out, h5py.File(in_path, "r") as f_in:
        # Total samples available in the file
        available = int(f_out["/Output1"].shape[0])

        # Use up to num_samples, but not more than available
        nsamples = min(num_samples, available)
        if nsamples <= 0:
            raise ValueError(f"No samples available in {out_path} (Output1 has size {available}).")

        for i in range(0, nsamples, chunk_size):
            end = min(i + chunk_size, nsamples)

            # Outputs
            Pred     = f_out["/Output1"][i:end]
            f_target = f_out["/Output2"][i:end]

            # Inputs
            Perm = f_in["/Input1"][i:end]
            Poro = f_in["/Input2"][i:end]
            vec  = f_in["/Input8"][i:end]   # already corrected by you

            # Branch: permeability & porosity
            branch = np.stack([Perm, Poro], axis=1)          # (batch, 2, H, W)
            branch_tensor = torch.from_numpy(branch).float()

            # Targets: stack pressure & front
            t1 = Pred.reshape(end - i, -1, 1)
            t2 = f_target.reshape(end - i, -1, 1)
            t1_tensor = torch.from_numpy(t1).float()
            t2_tensor = torch.from_numpy(t2).float()
            target = torch.cat([t1_tensor, t2_tensor], dim=2)  # (batch, S*T, 2)

            # Scalars: you said you already fixed this; using whole vec here
            scalars = torch.from_numpy(vec).float()

            branch_chunks.append(branch_tensor)
            target_chunks.append(target)
            scalar_chunks.append(scalars)

    # ----- concatenate all chunks -----
    branch_inputs    = torch.cat(branch_chunks, dim=0) if branch_chunks else torch.empty(0)
    targets_combined = torch.cat(target_chunks, dim=0) if target_chunks else torch.empty(0)
    scalars_combined = torch.cat(scalar_chunks, dim=0) if scalar_chunks else torch.empty(0)

    print("Loaded test samples:", branch_inputs.shape[0])
    print("branch:", branch_inputs.shape)

    return branch_inputs, targets_combined, scalars_combined, trunk_inputs_encoded


# data_utils.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int, rank: int = 0, deterministic: bool = True, scope: str = "all") -> None:
    """
    Seed helper with scoped control.

    scope:
      - "weights": seed only torch (for model init, dropout, etc.)
      - "data":    seed numpy/random + create a torch.Generator for DataLoader shuffling
      - "all":     do both
    """
    if scope in ("weights", "all"):
        wseed = seed + rank
        torch.manual_seed(wseed)
        torch.cuda.manual_seed(wseed)
        torch.cuda.manual_seed_all(wseed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if scope in ("data", "all"):
        dseed = seed  # no rank offset → same dataset split on every rank
        np.random.seed(dseed)
        random.seed(dseed)
        # Stash a base seed for workers & samplers via env (simple, import-free)
        os.environ["DATA_BASE_SEED"] = str(dseed)

    print(f"[Rank {rank}] set_seed(scope={scope}) -> "
          f"weights_seed={seed+rank if scope in ('weights','all') else '—'}, "
          f"data_seed={seed if scope in ('data','all') else '—'}, "
          f"deterministic={deterministic}")

def worker_init_fn(worker_id: int, base_seed: int = None):
    """
    Deterministic worker seeding. Uses DATA_BASE_SEED if base_seed is None.
    """
    if base_seed is None:
        base_seed = int(os.environ.get("DATA_BASE_SEED", "0"))
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
  
