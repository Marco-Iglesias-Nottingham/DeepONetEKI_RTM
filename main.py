import os
import argparse
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from scipy.io import savemat, loadmat

from emulator_tools.models import DeepONetPressureFront
from emulator_tools.data_utils import (
    generate_data,
    normalize_and_standardize,
    set_seed,
    worker_init_fn,
)
from emulator_tools.train import train, validate

from torch.optim.lr_scheduler import ReduceLROnPlateau


def make_lumped_mass_probs(mat_path: str, device: torch.device) -> torch.Tensor:
    """
    Load a (possibly sparse) MassMatrix from .mat and return a 1D probability
    vector [S] with row-sum lumping, normalized to sum == 1.
    """
    M = loadmat(mat_path)["MassMatrix"]
    m = np.array(M.sum(axis=1)).ravel().astype(np.float64)
    m = m / m.sum()
    return torch.tensor(m, dtype=torch.float32, device=device)


def setup_ddp():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    os.environ["MASTER_ADDR"] = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def run(args):
    logs = {
        "epoch": [],
        "train_total": [],
        "val_total": [],
        "time_50epoch_blocks": [],
    }


    output_dir = f"output_{args.out_name}"
    os.makedirs(output_dir, exist_ok=True)

    use_ddp = args.ddp
    if use_ddp:
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_TIMEOUT"] = "120"

        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Deterministic data
    set_seed(args.data_seed, rank=rank, deterministic=True, scope="data")

    # 2) Independently seed weights
    set_seed(args.weight_seed, rank=rank, deterministic=True, scope="weights")

    # Generate data
    num_samples = args.num_samples
    # 1) Load ALL data from disk
    branch_inputs, targets, scalar_inputs, trunk_inputs_encoded = generate_data()
    trunk_inputs_encoded = trunk_inputs_encoded.to(device)

    N_total = branch_inputs.shape[0]
    num_samples = args.num_samples

    if num_samples > N_total:
        raise ValueError(
            f"Requested num_samples={num_samples}, but only {N_total} are available in the file."
        )

    # 2) Randomly select 'num_samples' indices
    g = torch.Generator()
    g.manual_seed(args.data_seed)

    perm = torch.randperm(N_total, generator=g)
    selected = perm[:num_samples]

    branch_inputs = branch_inputs[selected]
    targets       = targets[selected]
    scalar_inputs = scalar_inputs[selected]
    N = num_samples  # for clarity below

    train_size = int(0.9 * N)
    val_size = int(0.1 * N)
    test_size = N - train_size - val_size

    train_branch, val_branch, test_branch = torch.split(branch_inputs, [train_size, val_size, test_size])
    train_scalars, val_scalars, test_scalars = torch.split(scalar_inputs, [train_size, val_size, test_size])
    train_targets, val_targets, test_targets = torch.split(targets, [train_size, val_size, test_size])

    # Normalization / standardization
    stats_file_path = os.path.join(output_dir, "normalisation_data.pt")

    (
        (train_branch_norm, val_branch_norm, test_branch_norm),
        (train_scalars_norm, val_scalars_norm, test_scalars_norm),
        (train_targets_norm, val_targets_norm, test_targets_norm),
        stats,
    ) = normalize_and_standardize(
        train_branch,
        val_branch,
        test_branch,
        train_scalars,
        val_scalars,
        test_scalars,
        train_targets,
        val_targets,
        test_targets,
        output_stats_file=stats_file_path,
    )

    p_mean = stats["target_mean"]
    p_std = stats["target_std"]

    # Save test data (only on rank 0)
    if rank == 0:
        test_data_dir = os.path.join(output_dir, "saved_test_data")
        os.makedirs(test_data_dir, exist_ok=True)
        torch.save(
            {
                "test_branch_norm": test_branch_norm,
                "test_scalars_norm": test_scalars_norm,
                "test_targets_norm": test_targets_norm,
                "trunk_inputs_encoded": trunk_inputs_encoded,
                "p_mean": p_mean,
                "p_std": p_std,
            },
            os.path.join(test_data_dir, "test_data.pt"),
        )

    # Datasets
    train_ds = TensorDataset(train_branch_norm, train_scalars_norm, train_targets_norm)
    val_ds = TensorDataset(val_branch_norm, val_scalars_norm, val_targets_norm)

    worker_init = lambda wid: worker_init_fn(wid + rank * 1000, base_seed=args.data_seed)

    if use_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.data_seed,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=args.data_seed,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=not use_ddp,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init,
        generator=None if use_ddp else g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=worker_init,
    )

    # Model
    trunk_input_dim = trunk_inputs_encoded.shape[-1]
    Dp = Df = max(1, int(args.D))

    model = DeepONetPressureFront(
        branch_input_channels=4,
        scalar_dim=5,
        trunk_input_dim=trunk_input_dim,
        Dp=Dp,
        Df=Df,
        num_layers=6,
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Rank {rank}] Model has {n_params:,} trainable parameters.")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-9)

    # Single scheduler: ReduceLROnPlateau
    if args.lr_schedule == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.75,
            patience=20,
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.lr_schedule}")

    if use_ddp:
        print(f"[Rank {rank}] Running on GPU {local_rank} (world size: {world_size})")

    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_TIMEOUT"] = "120"

    # Importance sampling weights
    mat_path = os.path.join("MATLAB_files_for_emulator", "MassMatrix.mat")
    lumped_probs = make_lumped_mass_probs(mat_path, device)
    assert lumped_probs.numel() == trunk_inputs_encoded.shape[0], (
        "MassMatrix nodes do not match trunk spatial size."
    )

    # Training loop
    start_time = time.perf_counter()
    for epoch in range(args.epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            trunk_inputs_encoded,
            p_mean,
            p_std,
            world_size=world_size,
            lumped_probs=lumped_probs,
            bce_weight=0.05,
        )

        val_loss = validate(
            model,
            val_loader,
            device,
            epoch,
            trunk_inputs_encoded,
            p_mean,
            p_std,
            world_size=world_size,
            num_space_points_val=500,
            lumped_probs=lumped_probs,
            bce_weight=0.05,
        )

        if rank == 0:
            log_template = (
                "Epoch {epoch:3d} {phase:<5} | "
                "Total: {total:7.5f} | "
                "p_loss: {p_loss:7.5f} | "
                "f_loss: {f_loss:7.5f} | "
                "LR: {lr:.4e}"
            )
            print(
                log_template.format(
                    epoch=epoch + 1,
                    phase="TRAIN",
                    total=train_loss["total"],
                    p_loss=train_loss["p_loss"],
                    f_loss=train_loss["f_loss"],
                    lr=current_lr,
                )
            )
            print(
                log_template.format(
                    epoch=epoch + 1,
                    phase="VAL",
                    total=val_loss["total"],
                    p_loss=val_loss["p_loss"],
                    f_loss=val_loss["f_loss"],
                    lr=current_lr,
                )
            )

        # Plateau scheduler uses validation metric
        scheduler.step(val_loss["total"])

        logs["epoch"].append(epoch + 1)
        logs["train_total"].append(train_loss["total"])
        logs["val_total"].append(val_loss["total"])
        
        if (epoch + 1) % 50 == 0 and rank == 0:
            elapsed = time.perf_counter() - start_time

            logs["time_50epoch_blocks"].append(float(elapsed))

            mat_dict = {
                "epoch": logs["epoch"],
                "train_total": logs["train_total"],
                "val_total": logs["val_total"],
                "time_50epoch_blocks": logs["time_50epoch_blocks"],  # ✅ now a vector
            }

            losses_path = os.path.join(output_dir, "losses_latest.mat")
            savemat(losses_path, mat_dict)

            model_path = os.path.join(output_dir, f"deeponet_epoch_{epoch + 1:03d}.pt")
            torch.save(
                model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                model_path,
            )

            print(f"[Rank {rank}] Saved losses and model to: {output_dir}")



    if use_ddp:
        cleanup_ddp()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddp", action="store_true", help="Enable DDP training")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--weight-seed", type=int, default=42, help="Seed for model init / torch RNG")
    parser.add_argument("--data-seed", type=int, default=0, help="Seed for any data shuffling / loaders")
    parser.add_argument(
        "--lr-schedule",
        choices=["plateau"],
        default="plateau",
        help="Learning rate scheduler (currently only 'plateau' is implemented).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=4e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="default",
        help="Suffix for output directory name, e.g., Exp1 → output_Exp1/",
    )
    parser.add_argument(
        "--D",
        type=int,
        default=400,
        help="Latent channel width used for BOTH heads (sets Dp = Df = D).",
    )

    args = parser.parse_args()

    start_time = time.perf_counter()
    run(args)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    print(
        f"\n[Total Runtime] {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)"
    )


if __name__ == "__main__":
    main()

