
# test_model.py
"""
Offline evaluation script for DeepONet emulator.

- Loads a trained model checkpoint.
- Loads / normalizes test data.
- Runs prediction in trunk-chunks to reduce GPU memory.
- Computes pressure and front errors, plus index-set statistics.
- Saves everything to .mat and .pkl for post-processing in MATLAB / Python.
"""

import os
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import loadmat, savemat

from emulator_tools.models import DeepONetPressureFront
from emulator_tools.data_utils import (
    add_positional_channels,
    generate_data_for_testing,
    apply_saved_normalization,
)

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def pick_test_batch_size(D: int) -> int:
    return 8 if D == 800 else 16


def to_numpy(t):
    """✅ SAFE conversion from CUDA / bfloat16 → float32 CPU numpy"""
    if isinstance(t, torch.Tensor):
        return t.detach().to(torch.float32).cpu().numpy()
    return np.asarray(t)


# ---------------------------------------------------------------------------
# Chunked prediction
# ---------------------------------------------------------------------------

def predict_chunked(model, branch_batch, scalar_batch, trunk_inputs_encoded, chunk_size=8192):
    B = branch_batch.size(0)
    Dtrunk = trunk_inputs_encoded.shape[-1]

    trunk_flat = trunk_inputs_encoded.view(-1, Dtrunk)

    p_chunks, f_chunks = [], []
    bb = add_positional_channels(branch_batch)

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for start in range(0, trunk_flat.size(0), chunk_size):
            stop = min(start + chunk_size, trunk_flat.size(0))
            t_slice = trunk_flat[start:stop].unsqueeze(0).expand(B, -1, -1).contiguous()
            p_chunk, f_chunk = model(bb, scalar_batch, t_slice)
            p_chunks.append(p_chunk)
            f_chunks.append(f_chunk)

    return torch.cat(p_chunks, dim=1), torch.cat(f_chunks, dim=1)


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_indices(path, var="ind"):
    data = loadmat(path)
    ind = np.squeeze(data[var]).astype(np.int64)
    return torch.tensor(ind, dtype=torch.long)


ind_sets = {
    "for_real": _load_indices("MATLAB_files_for_emulator/indices_for_real.mat", var="ind"),
    "100":      _load_indices("MATLAB_files_for_emulator/indices_100.mat",      var="ind"),
}


# ---------------------------------------------------------------------------
# Main test routine
# ---------------------------------------------------------------------------

def test(model, test_loader, device, epoch, trunk_inputs_encoded, p_mean, p_std, out_dir):
    p_mean = float(p_mean)
    p_std = float(p_std)

    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    M = loadmat(os.path.join("MATLAB_files_for_emulator", "MassMatrix.mat"))["MassMatrix"]
    M_torch = torch.tensor(M.toarray(), dtype=torch.float64, device=device)

    all_rel_l2 = []
    all_rel_l2_front = []
    all_rel_l1_front = []

    saved_p_preds = []
    saved_p_trues = []
    saved_f_preds = []
    saved_f_trues = []
    saved_branch = []
    saved_scalar_branch = []

    err_vectors = {name: [] for name in ind_sets.keys()}

    p_mean_d = torch.tensor(p_mean, dtype=torch.float64, device=device)
    p_std_d  = torch.tensor(p_std,  dtype=torch.float64, device=device)

    with torch.no_grad():
        for branch_batch, scalar_batch, target_batch in test_loader:
            branch_batch = branch_batch.to(device)
            scalar_batch = scalar_batch.to(device)
            target_batch = target_batch.to(device)

            B = branch_batch.size(0)
            n_space, n_time = trunk_inputs_encoded.shape[:2]

            target_batch = target_batch.view(B, n_space, n_time, 2)
            target_sub = target_batch.view(B, -1, 2)

            p_pred, f_logit = predict_chunked(
                model, branch_batch, scalar_batch, trunk_inputs_encoded
            )
            f_pred = torch.sigmoid(f_logit)

            p_true = target_sub[..., 0:1]
            f_true = target_sub[..., 1:2]

            f_true_bin = (f_true > 0.9).float()
            f_pred_bin = (f_pred > 0.9).float()

            p_pred = p_pred.view(B, n_space, n_time).transpose(1, 2)
            p_true = p_true.view(B, n_space, n_time).transpose(1, 2)
            f_pred_bin = f_pred_bin.view(B, n_space, n_time).transpose(1, 2)
            f_true_bin = f_true_bin.view(B, n_space, n_time).transpose(1, 2)

            # ✅ De-normalize
            p_true_un = p_true.double() * p_std_d + p_mean_d
            p_pred_un = (p_pred.double() * p_std_d + p_mean_d) * f_pred_bin.double()

            # ✅ Pressure error
            errors = p_pred_un - p_true_un
            errors_sq = torch.einsum("bts,ss,bts->bt", errors, M_torch, errors)
            rel_l2 = torch.sqrt(errors_sq.sum(dim=1))

            true_sq = torch.einsum("bts,ss,bts->bt", p_true_un, M_torch, p_true_un)
            norm_true = torch.sqrt(true_sq.sum(dim=1))
            all_rel_l2.append(rel_l2 / (norm_true + 1e-12))

            # ✅ Front L2
            diff = f_pred - f_true
            err_sq = torch.einsum("bts,ss,bts->bt", diff, M_torch, diff)
            rel_l2_front = torch.sqrt(err_sq.sum(dim=1))

            true_sq = torch.einsum("bts,ss,bts->bt", f_true, M_torch, f_true)
            denom = torch.sqrt(true_sq.sum(dim=1))
            all_rel_l2_front.append(rel_l2_front / (denom + 1e-12))

            # ✅ Front L1
            abs_diff = torch.abs(f_pred - f_true)
            num = torch.einsum("bts,ss,bts->bt", abs_diff, M_torch, torch.ones_like(abs_diff))
            denom = torch.einsum("bts,ss,bts->bt", f_true, M_torch, torch.ones_like(f_true))
            all_rel_l1_front.append(num.sum(dim=1) / (denom.sum(dim=1) + 1e-12))

            # ✅ Save one per batch (SAFE)
            saved_p_preds.append(to_numpy(p_pred_un[0]))
            saved_p_trues.append(to_numpy(p_true_un[0]))
            saved_f_preds.append(to_numpy(f_pred[0]))
            saved_f_trues.append(to_numpy(f_true[0]))
            saved_branch.append(to_numpy(branch_batch[0]))
            saved_scalar_branch.append(to_numpy(scalar_batch[0]))

            # ✅ Index-set errors
            for tag, inds in ind_sets.items():
                inds = inds.to(device)
                e = p_pred_un[:, :, inds] - p_true_un[:, :, inds]
                e = e.permute(0, 2, 1).reshape(B, -1)
                err_vectors[tag].append(e.cpu())

    # ✅ SAFE CPU aggregation
    rel_l2_all = torch.cat(all_rel_l2).cpu().numpy()
    rel_l2_front_all = torch.cat(all_rel_l2_front).cpu().numpy()
    rel_l1_front_all = torch.cat(all_rel_l1_front).cpu().numpy()

    metrics = {
        "rel_l2_mean": rel_l2_all.mean(),
        "rel_l2_std":  rel_l2_all.std(),
        "rel_l2_front_mean": rel_l2_front_all.mean(),
        "rel_l2_front_std":  rel_l2_front_all.std(),
        "rel_l1_front_mean": rel_l1_front_all.mean(),
        "rel_l1_front_std":  rel_l1_front_all.std(),
    }

    indexset_stats = {}
    for tag, chunks in err_vectors.items():
        E = torch.cat(chunks)
        mean_err = E.mean(dim=0)
        Xc = E - mean_err
        cov = (Xc.T @ Xc) / max(E.shape[0] - 1, 1)
        indexset_stats[tag] = {
            "mean_error": mean_err.cpu().numpy(),
            "sample_cov": cov.cpu().numpy(),
        }

    results_dict = {
        **metrics,
        "p_preds_samples": np.array(saved_p_preds),
        "p_trues_samples": np.array(saved_p_trues),
        "f_preds_samples": np.array(saved_f_preds),
        "f_trues_samples": np.array(saved_f_trues),
        "branch_samples": np.array(saved_branch),
        "scalar_branch_samples": np.array(saved_scalar_branch),

        "for_real_mean_error": indexset_stats["for_real"]["mean_error"],
        "for_real_sample_cov": indexset_stats["for_real"]["sample_cov"],
        "idx100_mean_error":   indexset_stats["100"]["mean_error"],
        "idx100_sample_cov":   indexset_stats["100"]["sample_cov"],
    }

    mat_path = os.path.join(out_dir, f"errors_and_samples_epoch_{epoch:03d}.mat")
    pkl_path = os.path.join(out_dir, f"errors_and_samples_epoch_{epoch:03d}.pkl")

    savemat(mat_path, results_dict)
    with open(pkl_path, "wb") as f:
        pickle.dump(results_dict, f)

    print(f"✅ Saved:\n  → {mat_path}\n  → {pkl_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-name", type=str, default="Hard")
    parser.add_argument("--epoch", type=int, default=600)
    parser.add_argument("--D", type=int, default=400)
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = f"output_{args.out_name}"
    epoch_str = f"{args.epoch:03d}"
    model_path = os.path.join(output_dir, f"deeponet_epoch_{epoch_str}.pt")
    stats_path = os.path.join(output_dir, "normalisation_data.pt")
    test_output_dir = os.path.join(output_dir, "test_outputs_epoch")

    os.makedirs(test_output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_samples = 10000
    branch_inputs, targets, scalar_inputs, trunk_inputs_encoded = generate_data_for_testing(num_samples)

    stats = torch.load(stats_path)
    p_mean, p_std = stats["target_mean"], stats["target_std"]

    test_branch_norm, test_scalars_norm, test_targets_norm = apply_saved_normalization(
        branch_inputs, scalar_inputs, targets, stats
    )

    trunk_inputs_encoded = trunk_inputs_encoded.to(device)

    test_ds = TensorDataset(test_branch_norm, test_scalars_norm, test_targets_norm)
    test_loader = DataLoader(
        test_ds,
        batch_size=pick_test_batch_size(args.D),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model = DeepONetPressureFront(
        branch_input_channels=4,
        scalar_dim=5,
        trunk_input_dim=trunk_inputs_encoded.shape[-1],
        Dp=args.D,
        Df=args.D,
        num_layers=6,
    ).to(device)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    test(
        model,
        test_loader,
        device,
        args.epoch,
        trunk_inputs_encoded,
        p_mean,
        p_std,
        test_output_dir,
    )


if __name__ == "__main__":
    main()
