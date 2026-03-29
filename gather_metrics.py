#!/usr/bin/env python3
import os
import re
import glob
import argparse

import numpy as np
import h5py
import scipy.io as sio


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Aggregate test metrics across epochs.")
    p.add_argument(
        "-b",
        "--base-dir",
        required=True,
        help=(
            "Base experiment folder, e.g. output_NS40000_D400. "
            "This should contain the per-epoch test output subfolder."
        ),
    )
    p.add_argument(
        "--in-subdir",
        default="test_outputs_epoch",
        help="Subdirectory inside base-dir where the epoch files live "
             "(default: test_outputs_epoch).",
    )
    p.add_argument(
        "-o",
        "--out",
        default=None,
        help=(
            "Full path to output .mat file. "
            "Defaults to <base-dir>/aggregated_metrics.mat."
        ),
    )
    return p.parse_args()


# ------------- helpers --------------
METRIC_KEYS = [
    "rel_l2_mean",
    "rel_l2_std",
    "rel_l2_front_mean",
    "rel_l2_front_std",
    "rel_l1_front_mean",
    "rel_l1_front_std",
]


def read_scalar_h5(f, key):
    """Read a scalar metric from an HDF5 file, or NaN if missing/empty."""
    if key not in f:
        return np.nan
    arr = np.array(f[key][()]).squeeze()
    if arr.size == 0:
        return np.nan
    return float(arr) if arr.ndim == 0 else float(np.mean(arr))


def diag_l2_norm_h5(f, key):
    """
    L2 norm of the diagonal of a covariance matrix stored in HDF5.
    Returns NaN if missing or not square.
    """
    if key not in f:
        return np.nan
    cov = np.array(f[key][()])
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return np.nan
    return float(np.linalg.norm(np.diag(cov), ord=2))


def vec_l2_norm_h5(f, key):
    """L2 norm of a vector stored in HDF5 (flattened)."""
    if key not in f:
        return np.nan
    v = np.array(f[key][()]).squeeze()
    if v.size == 0:
        return np.nan
    return float(np.linalg.norm(v.ravel(), ord=2))


def read_scalar_mat(md, key):
    """Same as read_scalar_h5 but for a loaded .mat dict."""
    if key not in md:
        return np.nan
    arr = np.array(md[key]).squeeze()
    if arr.size == 0:
        return np.nan
    return float(arr) if arr.ndim == 0 else float(np.mean(arr))


def diag_l2_norm_mat(md, key):
    """L2 norm of diagonal of a covariance matrix in a .mat dict."""
    cov = md.get(key, None)
    if cov is None:
        return np.nan
    cov = np.array(cov)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return np.nan
    return float(np.linalg.norm(np.diag(cov), ord=2))


def vec_l2_norm_mat(md, key):
    """L2 norm of a vector in a .mat dict."""
    v = md.get(key, None)
    if v is None:
        return np.nan
    v = np.array(v).squeeze()
    if v.size == 0:
        return np.nan
    return float(np.linalg.norm(v.ravel(), ord=2))


# ------------- main ------------------
def main():
    args = parse_args()

    base_dir = args.base_dir
    in_dir = os.path.join(base_dir, args.in_subdir)
    out_path = args.out or os.path.join(base_dir, "aggregated_metrics.mat")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Look for per-epoch files like errors_and_samples_epoch_050.mat
    paths = sorted(glob.glob(os.path.join(in_dir, "errors_and_samples_epoch_*.mat")))
    epoch_re = re.compile(r"errors_and_samples_epoch_(\d+)\.mat$")

    epochs = []
    rows = {k: [] for k in METRIC_KEYS}
    rows["for_real_var_diag_l2"] = []
    rows["idx100_var_diag_l2"] = []
    rows["for_real_mean_err_l2"] = []
    rows["idx100_mean_err_l2"] = []

    for p in paths:
        m = epoch_re.search(os.path.basename(p))
        if not m:
            continue
        epoch = int(m.group(1))

        # Try HDF5 first; fall back to MATLAB .mat
        try:
            with h5py.File(p, "r") as f:
                vals = {k: read_scalar_h5(f, k) for k in METRIC_KEYS}
                fr_cov_l2 = diag_l2_norm_h5(f, "for_real_sample_cov")
                idx_cov_l2 = diag_l2_norm_h5(f, "idx100_sample_cov")
                fr_mean_l2 = vec_l2_norm_h5(f, "for_real_mean_error")
                idx_mean_l2 = vec_l2_norm_h5(f, "idx100_mean_error")
        except OSError:
            md = sio.loadmat(p, squeeze_me=True, struct_as_record=False)
            vals = {k: read_scalar_mat(md, k) for k in METRIC_KEYS}
            fr_cov_l2 = diag_l2_norm_mat(md, "for_real_sample_cov")
            idx_cov_l2 = diag_l2_norm_mat(md, "idx100_sample_cov")
            fr_mean_l2 = vec_l2_norm_mat(md, "for_real_mean_error")
            idx_mean_l2 = vec_l2_norm_mat(md, "idx100_mean_error")

        epochs.append(epoch)
        for k, v in vals.items():
            rows[k].append(v)
        rows["for_real_var_diag_l2"].append(fr_cov_l2)
        rows["idx100_var_diag_l2"].append(idx_cov_l2)
        rows["for_real_mean_err_l2"].append(fr_mean_l2)
        rows["idx100_mean_err_l2"].append(idx_mean_l2)

    if not epochs:
        print(f"⚠ No epoch files found in {in_dir}")
        return

    # Sort by epoch
    order = np.argsort(epochs)
    epochs_arr = np.array(epochs, dtype=np.int32)[order]
    for k in rows:
        rows[k] = np.array(rows[k], dtype=np.float64)[order]

    out = {"epochs": epochs_arr}
    out.update(rows)
    sio.savemat(out_path, out)

    print(f"✅ Saved aggregated metrics to: {out_path}")
    print("Fields:", list(out.keys()))


if __name__ == "__main__":
    main()

