import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import pickle
import scipy.io as sio

def load_results_pkl(path):
    """Load results from a .pkl file"""
    with open(path, "rb") as f:
        results = pickle.load(f)
    return results

def load_results_mat(path):
    """Load results from a .mat file"""
    results = sio.loadmat(path)
    # scipy.io.loadmat wraps keys in MATLAB style; filter out meta keys
    results = {k: v for k, v in results.items() if not k.startswith("__")}
    return results


def _to_numpy(a):
    """Accepts numpy arrays or torch tensors (cpu/gpu) and returns a numpy array."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

import matplotlib.tri as mtri

from pathlib import Path

def plot_three_rows_unstructured(p_true_TS, p_pred_TS, nodes_2S,
                                 times_idx=None, save_path=None,
                                 dpi=200, transparent=False):

    # ... your existing plotting code that builds `fig` ...


    if nodes_2S.shape[0] != 2 and nodes_2S.shape[1] == 2:
        nodes_2S = nodes_2S.T
    x, y = nodes_2S[0], nodes_2S[1]

    T, S = p_true_TS.shape
    times_idx = [t for t in times_idx if 0 <= t < T]

    tri = mtri.Triangulation(x, y)

    err_TS = np.abs(p_pred_TS - p_true_TS)

    # shared scale for p_true/p_pred over the selected times
    vmin_p = min(np.min(p_true_TS[times_idx]), np.min(p_pred_TS[times_idx]))
    vmax_p = max(np.max(p_true_TS[times_idx]), np.max(p_pred_TS[times_idx]))

    # separate scale for |error|
    err_sel = err_TS[times_idx]
    err_vmax = np.percentile(err_sel, 99) if np.any(err_sel) else 1.0

    n_cols = len(times_idx)
    fig = plt.figure(figsize=(3.1*n_cols + 1.2, 8.5))
    gs = fig.add_gridspec(3, n_cols + 1, width_ratios=[1]*n_cols + [0.05], hspace=0.15, wspace=0.08)

    # row 1: p_true
    im_true_ref = None
    for j, t in enumerate(times_idx):
        ax = fig.add_subplot(gs[0, j])
        im_true_ref = ax.tripcolor(tri, p_true_TS[t], vmin=vmin_p, vmax=vmax_p, shading='flat')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {t}")
    cax0 = fig.add_subplot(gs[0, -1])
    cb0 = fig.colorbar(im_true_ref, cax=cax0)
    cb0.set_label("p (shared)")

    # row 2: p_pred (same scale)
    im_pred_ref = None
    for j, t in enumerate(times_idx):
        ax = fig.add_subplot(gs[1, j])
        im_pred_ref = ax.tripcolor(tri, p_pred_TS[t], vmin=vmin_p, vmax=vmax_p, shading='flat')
        ax.set_xticks([]); ax.set_yticks([])
    cax1 = fig.add_subplot(gs[1, -1])
    plt.colorbar(im_pred_ref, cax=cax1)

    # row 3: |error|
    im_err_ref = None
    for j, t in enumerate(times_idx):
        ax = fig.add_subplot(gs[2, j])
        im_err_ref = ax.tripcolor(tri, err_TS[t], vmin=0.0, vmax=err_vmax, shading='flat')
        ax.set_xticks([]); ax.set_yticks([])
    cax2 = fig.add_subplot(gs[2, -1])
    cb2 = fig.colorbar(im_err_ref, cax=cax2)
    cb2.set_label("|p_pred - p_true|")

    # row labels
    fig.text(0.02, 0.86, "p_true", rotation=90, va='center', fontsize=11)
    fig.text(0.02, 0.54, "p_pred", rotation=90, va='center', fontsize=11)
    fig.text(0.02, 0.22, "|error|", rotation=90, va='center', fontsize=11)

    plt.show()

    # ---- save (optional) ----
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                    facecolor='white', transparent=transparent)
    plt.close(fig)  # free memory



def _to_numpy(a):
    try:
        import torch
        if hasattr(a, "detach"):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(a)

def _std_to_3d(A):
    """Make A into shape (Nsamples, Ntime, Npoints)."""
    A = _to_numpy(A)
    if A.ndim == 1:        # (Npoints,)
        A = A[None, None, :]
    elif A.ndim == 2:      # (Ntime, Npoints)
        A = A[None, ...]
    assert A.ndim == 3, f"Expected 3D (Nsamples,Ntime,Npoints), got {A.shape}"
    return A

import matplotlib.tri as mtri


def plot_three_rows_tri(
    p_true, p_pred, coord_matrix,
    sample_idx=0, time_indices=None,
    cmap="jet", save_path=None, dpi=300, panel_size=(3.8, 3.8),
    shading="gouraud",                    # "flat" for crisp triangles
    # refinement & triangle quality
    use_refiner=False,
    refiner_subdivisions=3,
    min_circle_ratio=0.0,
    # typography
    title_size=20, label_size=20, tick_size=11,
    cbar_label_size=14, cbar_tick_size=13,
    # layout controls
    wspace=0.25, hspace=0.05,
    cbar_col_ratio=0.06,
    # NEW: suptitle controls
    suptitle=None,
    suptitle_size=22,
    suptitle_y=0.98,
    suptitle_top=0.93,
    # NEW: robust error color scale controls
    error_clip_percentile=None,   # e.g., 99 or 99.5; if set, vmax_e is that percentile
    error_drop_k=0                # e.g., 2; drop the top-k absolute errors before setting vmax_e
):
    """Three rows: p_true (top), p_pred (middle), |p_pred - p_true| (bottom).
    Columns are selected time indices. Colorbars live in a dedicated rightmost column,
    so data panels remain equal widths.
    """
    # ---------- helpers ----------
    import numpy as np
    import matplotlib.pyplot as plt

    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["font.family"] = "serif"   # looks better with LaTeX
    def _to_numpy(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                return a.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(a)

    def _std_to_3d(a):
        A = _to_numpy(a)
        if A.ndim == 1:      # (Npoints,)
            A = A[None, None, :]
        elif A.ndim == 2:    # (Ntime, Npoints)
            A = A[None, ...]
        return A

    # ---------- data ----------
    Ptrue = _std_to_3d(p_true)
    Ppred = _std_to_3d(p_pred)
    assert Ptrue.shape == Ppred.shape, f"Shape mismatch: {Ptrue.shape} vs {Ppred.shape}"

    C = _to_numpy(coord_matrix)
    if C.shape[0] != 2 and C.shape[1] == 2:
        C = C.T
    assert C.shape[0] == 2, f"coord_matrix must be (2, Npoints), got {C.shape}"
    x, y = C[0], C[1]

    import matplotlib.tri as mtri
    tri = mtri.Triangulation(x, y)
    if min_circle_ratio and min_circle_ratio > 0:
        analyzer = mtri.TriAnalyzer(tri)
        tri.set_mask(analyzer.get_flat_tri_mask(min_circle_ratio))

    n_time = Ptrue.shape[1]
    if time_indices is None:
        time_indices = np.linspace(0, n_time - 1, 5, dtype=int)
    else:
        time_indices = np.asarray(time_indices, dtype=int)

    # global color scales (top+middle share; bottom uses error)
    vals_p = np.concatenate([
        Ptrue[sample_idx, time_indices, :].ravel(),
        Ppred[sample_idx, time_indices, :].ravel()
    ])
    vals_p = vals_p[np.isfinite(vals_p)]
    if vals_p.size:
        vmin_p = float(np.nanmin(vals_p))
        vmax_p = float(np.nanmax(vals_p))
        if not np.isfinite(vmin_p) or not np.isfinite(vmax_p) or np.isclose(vmin_p, vmax_p):
            vmax_p = vmin_p + 1e-12
    else:
        vmin_p, vmax_p = 0.0, 1.0

    Err = np.abs(Ppred - Ptrue)
    vals_e = Err[sample_idx, time_indices, :].ravel()
    vals_e = vals_e[np.isfinite(vals_e)]

    # ---------- NEW: robust vmax for error scale ----------
    if vals_e.size:
        vmin_e = float(np.nanmin(vals_e))

        # Sort once for both strategies
        vals_sorted = np.sort(vals_e)

        if error_clip_percentile is not None:
            # Use percentile (e.g., 99 or 99.5)
            p = float(error_clip_percentile)
            p = min(max(p, 0.0), 100.0)  # clamp
            vmax_e = float(np.percentile(vals_sorted, p))
        elif error_drop_k and error_drop_k > 0:
            k = int(error_drop_k)
            if vals_sorted.size > k:
                vmax_e = float(vals_sorted[-(k + 1)])  # the largest remaining after dropping top-k
            else:
                vmax_e = float(vals_sorted[-1])        # not enough values; fall back to max
        else:
            vmax_e = float(vals_sorted[-1])            # original behavior

        # Numerical safety
        if not np.isfinite(vmax_e) or np.isclose(vmin_e, vmax_e):
            vmax_e = vmin_e + 1e-12
    else:
        vmin_e, vmax_e = 0.0, 1.0

    # ---------- layout: data grid + dedicated colorbar column ----------
    ncols = len(time_indices) 
    nrows = 3

    # total figure size (add one narrow colorbar column)
    fig_w = panel_size[0] * (ncols + cbar_col_ratio)
    fig_h = panel_size[1] * nrows
    fig = plt.figure(figsize=(fig_w, fig_h))

    # width ratios: ncols data columns of 1, plus one cbar column of cbar_col_ratio
    wr = [1.0] * ncols + [cbar_col_ratio]
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=wr, wspace=wspace, hspace=hspace)

    # NEW: add super title and reserve top margin for it
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_size, y=suptitle_y)
        gs.update(top=suptitle_top)

    # data axes
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r, c])

    # colorbar axes (rightmost column)
    cax_p = fig.add_subplot(gs[0:2, -1])   # shared for top+middle
    cax_e = fig.add_subplot(gs[2,   -1])   # bottom (error)

    # optional refiner
    refiner = mtri.UniformTriRefiner(tri) if use_refiner else None

    def _tripcolor(ax, z, vmin, vmax):
        if use_refiner:
            tri_r, z_r = refiner.refine_field(z, subdiv=refiner_subdivisions)
            pc = ax.tripcolor(tri_r, z_r, shading=shading, cmap=cmap,
                              vmin=vmin, vmax=vmax, edgecolors="none", antialiased=False, linewidth=0)
        else:
            pc = ax.tripcolor(tri, z, shading=shading, cmap=cmap,
                              vmin=vmin, vmax=vmax, edgecolors="none", antialiased=False, linewidth=0)
        return pc

    # ---------- plot rows ----------
    last_pc_pred = None
    last_pc_err  = None

    # top: p_true
    for j, t in enumerate(time_indices):
        ax = axes[0, j]
        z = Ptrue[sample_idx, t, :]
        _tripcolor(ax, z, vmin_p, vmax_p)
        ax.set_title(f"t={int(t+1)}", fontsize=title_size)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.margins(0); ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=tick_size)

    # middle: p_pred
    for j, t in enumerate(time_indices):
        ax = axes[1, j]
        z = Ppred[sample_idx, t, :]
        last_pc_pred = _tripcolor(ax, z, vmin_p, vmax_p)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.margins(0); ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=tick_size)

    # bottom: |error|
    for j, t in enumerate(time_indices):
        ax = axes[2, j]
        z = Err[sample_idx, t, :]
        last_pc_err = _tripcolor(ax, z, vmin_e, vmax_e)
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.set_ylim(float(np.min(y)), float(np.max(y)))
        ax.margins(0); ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=tick_size)

    # left-side row labels
    axes[0, 0].set_ylabel("Target", fontsize=label_size)
    axes[1, 0].set_ylabel("Prediction", fontsize=label_size)
    axes[2, 0].set_ylabel("Absolute Error", fontsize=label_size)

    # colorbars
    cb_p = fig.colorbar(last_pc_pred, cax=cax_p)
    cb_p.ax.tick_params(labelsize=cbar_tick_size)

    cb_e = fig.colorbar(last_pc_err, cax=cax_e)
    cb_e.ax.tick_params(labelsize=cbar_tick_size)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Load from pickle
    pkl_results = load_results_pkl("output_data/errors_and_samples_epoch_450.pkl")
    #pkl_results = load_results_pkl("./output_Final2/test_outputs_epoch/errors_and_samples_epoch_300.pkl")
    print("Keys in .pkl:", list(pkl_results.keys()))
    print("rel_l2_mean =", pkl_results["rel_l2_mean"])
    print("branch_samples shape =", pkl_results["branch_samples"].shape)
    p_pred=pkl_results["p_preds_samples"]
    p_true=pkl_results["p_trues_samples"]
    f_pred=pkl_results["f_preds_samples"]
    f_true=pkl_results["f_trues_samples"]
    Branch=pkl_results["branch_samples"]
    Scalar=pkl_results["scalar_branch_samples"]

    B, T, S = 625, 34, 2973

    # f_pred: (B, S*T, 1)
    f_pred = (
        f_pred.squeeze(-1)          # (B, S*T)
              .reshape(B, S, T)     # (B, S, T)  with T fastest
              .transpose(0, 2, 1)   # (B, T, S) -> (625, 34, 2973)
    )
    # f_pred: (B, S*T, 1)
    f_true = (
        f_true.squeeze(-1)          # (B, S*T)
              .reshape(B, S, T)     # (B, S, T)  with T fastest
              .transpose(0, 2, 1)   # (B, T, S) -> (625, 34, 2973)
    )

    
    stats_path = "output_data/normalisation_data.pt"
    stats = torch.load(stats_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    branch_min = torch.tensor(stats["branch_min"], dtype=torch.float64, device=device)
    branch_max = torch.tensor(stats["branch_max"], dtype=torch.float64, device=device)
    scalar_mean = torch.tensor(stats["scalar_mean"], dtype=torch.float64, device=device)
    scalar_std  = torch.tensor(stats["scalar_std"],  dtype=torch.float64, device=device)
    p_mean      = torch.tensor(stats["target_mean"], dtype=torch.float64, device=device)
    p_std       = torch.tensor(stats["target_std"],  dtype=torch.float64, device=device)
    
    
    times=[0,5,10,20,30]
    # Choose one saved sample (index 0).
    # p_pred_samples/p_true_samples are typically [N_saved, T, S]
    idx=2
    p_pred0 = p_pred[idx]  # [T, S]
    p_true0 = p_true[idx]  # [T, S]
    f_pred0 = f_pred[idx].reshape(34,2973)  # [T, S]
    f_true0 = f_true[idx].reshape(34,2973)  # [T, S]
    
    mat = sio.loadmat('../MATLAB_files_for_emulator/Nodes.mat')
    coord_matrix = mat['Nodes']
    coords = coord_matrix
    if coords.shape[0] != 2 and coords.shape[1] == 2:
        coords = coords.T
    

    import matplotlib as mpl
    mpl.rcParams["text.usetex"] = True
    
    sample_idx=23  #19, 21
    # p_true, p_pred: (Nsamples, Ntime, Npoints)
    # coords: (2, Npoints)
    
    plot_three_rows_tri(
        p_true, p_pred, coord_matrix,
        sample_idx=sample_idx, time_indices=[0, 9, 19, 26, 30],
        cmap="jet",
        save_path="Figures/pressure_emulator.png", dpi=600,
        suptitle=r'$p(x,y,t)$',
   suptitle_size=22, suptitle_y=0.985, suptitle_top=0.93,
   error_clip_percentile=99
    )
    
    plot_three_rows_tri(
        f_true, f_pred, coord_matrix,
        sample_idx=sample_idx, time_indices=[0, 9, 19, 26, 30],
        cmap="gray",
        save_path="Figures/front_emulator.png", dpi=600,
        suptitle=r'$f(x,y,t)$',
   suptitle_size=22, suptitle_y=0.985, suptitle_top=0.93
    )
    

    extent = [0, 0.3, 0, 0.3]  # [xmin, xmax, ymin, ymax]

    # --- to CPU scalars for convenience ---
    bmin0 = float(branch_min[0].cpu()); bmax0 = float(branch_max[0].cpu())
    bmin1 = float(branch_min[1].cpu()); bmax1 = float(branch_max[1].cpu())
    
    # normalized fields (assumes channel 0 = log-permeability, 1 = porosity)
    perm_norm = Branch[sample_idx, 0, :, :]   # (120, 120)
    poro_norm = Branch[sample_idx, 1, :, :]   # (120, 120)
    Scalar = torch.tensor(Scalar, dtype=torch.float64, device=device)

    scalar_denorm=Scalar*scalar_std+scalar_mean
    print(scalar_denorm[sample_idx,:])
    # denormalize: X = X_norm*(max-min) + min
    perm = perm_norm * (bmax0 - bmin0) + bmin0  # interpreted as log K
    poro = poro_norm * (bmax1 - bmin1) + bmin1  # porosity phi
    
    perm_plot = np.asarray(perm)
    poro_plot = np.asarray(poro)
    
    # --- plot side-by-side with LaTeX titles/labels ---
    plt.rcParams["axes.grid"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    
    im0 = axes[0].imshow(perm_plot, origin="lower", cmap="jet", aspect="equal", extent=extent)
    axes[0].set_title(r"$\log K$", fontsize=16)
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    cbar0 = plt.colorbar(im0, ax=axes[0], shrink=1.0, aspect=25, pad=0.02)
    #cbar0.set_label(r"$\log K$")

    
    im1 = axes[1].imshow(poro_plot, origin="lower", cmap="jet", aspect="equal", extent=extent)
    axes[1].set_title(r"$\phi$", fontsize=16)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=1.0, aspect=25, pad=0.02)
#    cbar1.set_label(r"$\phi$")
    
    fig.savefig("Figures/poro_perm.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
   