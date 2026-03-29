# DeepONet Training for the Resin-Infusion Emulator

This repository contains the Python training pipeline used to train a `DeepONetPressureFront` model for the resin-infusion emulator. The training code supports both single-GPU execution and distributed multi-GPU training through PyTorch Distributed Data Parallel (DDP) in a SLURM environment.

The workflow:
- loads emulator data generated from the MATLAB resin-infusion code,
- constructs branch, scalar, target, and trunk inputs,
- randomly selects a subset of samples,
- splits the data into training and validation sets (with a small internal test split),
- normalizes and standardizes the inputs and outputs,
- trains a DeepONet model to predict pressure and filling-factor outputs,
- validates performance during training,
- saves normalization statistics, loss histories, and model checkpoints.

---

## Main Files

The main Python training script is assumed to be:

```text
main.py
```

It uses functionality from:

- `emulator_tools.models`
- `emulator_tools.data_utils`
- `emulator_tools.train`

In particular, the script imports:
- `DeepONetPressureFront`
- `generate_data`
- `normalize_and_standardize`
- `set_seed`
- `worker_init_fn`
- `train`
- `validate`

---

## Overview of the Training Procedure

The model uses three types of inputs:

- **Branch inputs**: spatially varying fields
- **Scalar inputs**: auxiliary scalar parameters
- **Trunk inputs**: encoded spatial-temporal coordinates

The script:
1. sets deterministic seeds for reproducibility,
2. loads all available training data from disk,
3. randomly selects `num_samples` samples,
4. splits the selected data into training, validation, and internal test subsets,
5. normalizes and standardizes the selected data,
6. constructs the DeepONet model,
7. trains the model using `AdamW`,
8. adapts the learning rate using `ReduceLROnPlateau`,
9. periodically saves loss histories and model checkpoints.

A lumped mass matrix derived from a MATLAB file is used for spatial weighting during training and validation.

---

## Dependencies

### Python packages

The code depends on:

- `torch`
- `numpy`
- `scipy`
- `h5py`

and a local Python package:

- `emulator_tools`



You must ensure that `emulator_tools` is available in the Python path.

---

## Dependencies on MATLAB-Generated Files

The Python training code depends directly on files produced by the MATLAB workflow.

### Required MATLAB files

The following MATLAB-derived files are required:

- `MATLAB_files_for_emulator/Nodes.mat`
- `MATLAB_files_for_emulator/MassMatrix.mat`

#### `Nodes.mat`

This file is loaded in `generate_data()`:

```python
mat = sio.loadmat("MATLAB_files_for_emulator/Nodes.mat")
coord_matrix = torch.tensor(mat["Nodes"], dtype=torch.float32) / 0.3
```

It provides the spatial node coordinates used to construct the trunk input. These coordinates are combined with the hardcoded time instances to form the spatial-temporal grid for the trunk network.

#### `MassMatrix.mat`

This file is loaded in:

```python
make_lumped_mass_probs(mat_path, device)
```

and used to construct lumped-mass probabilities by summing the rows of `MassMatrix`. These weights are then used in training and validation.

The main script expects this file at:

```text
MATLAB_files_for_emulator/MassMatrix.mat
```

---

## Dependencies on MATLAB-Generated HDF5 Data

The training data are expected to come from HDF5 files created from the MATLAB emulator workflow.

### Training data files

`generate_data()` currently reads from the fixed file pair:

```text
input_data_batch_seed_1.h5
output_data_batch_seed_1.h5
```

These filenames are hardcoded in `data_utils.py`.

---

## Expected HDF5 Contents

The code expects the following datasets.

### Input file

The input HDF5 file must contain:

- `/Input1`: permeability field
- `/Input2`: porosity field
- `/Input8`: scalar parameters

### Output file

The output HDF5 file must contain:

- `/Output1`: pressure output
- `/Output2`: filling-factor output

---

## How the Data Are Used

### Branch inputs

In `generate_data()`, the branch input is currently constructed as:

```python
branch = np.stack([Perm, Poro], axis=1)
```

so the present implementation uses:
- permeability from `Input1`
- porosity from `Input2`

This produces a branch tensor of shape:

```text
[N, 2, H, W]
```

### Scalar inputs

The scalar input is taken from `Input8`.

In `generate_data()`, only the first five entries are used:

```python
scalars = torch.from_numpy(vec[:, :5]).float()
```

so the current training setup assumes five scalar inputs.

---

## Trunk Input Construction

The trunk input is built from:
- node coordinates from `Nodes.mat`,
- a hardcoded list of time instances.

The time points are:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
17, 19, 21, 23, 25, 27, 30, 35, 40, 45, 50, 55, 60,
65, 70, 80, 90, 100, 110
```

These are normalized to `[0,1]` and then combined with the node coordinates to form the trunk grid. A Fourier encoding is then applied.

Thus, the trunk input depends on:
- the contents of `Nodes.mat`,
- the hardcoded time discretization in `data_utils.py`.

---

## Normalization and Data Splitting

During training, the dataset is:
- randomly subsampled,
- split into training and validation sets,
- with a small internal test split also created by the script.

The normalization and standardization procedure is applied as follows:

- **Branch inputs**: min–max normalization (per channel)
- **Scalar inputs**: standardization (zero mean, unit variance)
- **Pressure output**: standardization
- **Filling factor**: left unnormalized

The normalization statistics are computed using the training set only and saved to:

```text
output_<out_name>/normalisation_data.pt
```

This file contains:
- `branch_min`, `branch_max`
- `scalar_mean`, `scalar_std`
- `target_mean`, `target_std`

These statistics are required for any downstream use of the trained model, including inference and evaluation, and must be reused to ensure consistency between training and testing.

---

## Note on Testing Workflow

Although the training script creates and saves an internal test split, this is **not used for the main evaluation results**.

All reported testing and evaluation are performed using a **separate testing script**, which:
- loads the trained model,
- applies the saved normalization statistics from `normalisation_data.pt`,
- evaluates on independently generated or provided test datasets.

The internal test data saved during training are therefore **not part of the main evaluation pipeline** and are not required to reproduce the reported testing results.

---

## Reproducibility

Two separate random seeds are used:

- `data_seed`: controls sample selection, data shuffling, and worker seeds
- `weight_seed`: controls model initialization and torch random state for weights

This allows the dataset split and the network initialization to be reproduced independently.

In the SLURM batch script used for the reported runs, the values were:

```text
data-seed   = 529
weight-seed = 1234
```

---

## Model Definition

The training script constructs the model as:

```python
DeepONetPressureFront(
    branch_input_channels=4,
    scalar_dim=5,
    trunk_input_dim=trunk_input_dim,
    Dp=Dp,
    Df=Df,
    num_layers=6,
)
```

where:
- `scalar_dim = 5`
- `Dp = Df = D`
- `num_layers = 6`

### Important note

Based on the current `generate_data()` implementation, the branch input is built from only two channels:

- permeability
- porosity

That is, the data loader currently creates tensors of shape:

```text
[N, 2, H, W]
```

whereas the model is instantiated with:

```python
branch_input_channels=4
```

This means there is a potential implementation mismatch between the data-loading code and the model definition. For full reproducibility, these two parts of the code should be checked and made consistent.

---

## Command-Line Arguments

The training script supports the following command-line options.

### `--ddp`

Enable distributed training using PyTorch DDP.

```bash
--ddp
```

Default: disabled

---

### `--num-samples`

Number of samples randomly selected from the full dataset.

```bash
--num-samples 10000
```

Default:

```text
10000
```

---

### `--batch-size`

Mini-batch size used in training and validation.

```bash
--batch-size 32
```

Default:

```text
32
```

---

### `--epochs`

Number of training epochs.

```bash
--epochs 300
```

Default:

```text
300
```

---

### `--weight-seed`

Seed for model initialization and weight-related random state.

```bash
--weight-seed 42
```

Default:

```text
42
```

---

### `--data-seed`

Seed for sample selection, data shuffling, and data-loader reproducibility.

```bash
--data-seed 0
```

Default:

```text
0
```

---

### `--lr-schedule`

Learning-rate schedule. Currently only plateau scheduling is implemented.

```bash
--lr-schedule plateau
```

Default:

```text
plateau
```

---

### `--lr`

Initial learning rate.

```bash
--lr 4e-4
```

Default:

```text
4e-4
```

---

### `--out-name`

Suffix used to name the output folder.

```bash
--out-name Exp1
```

This produces:

```text
output_Exp1/
```

Default:

```text
default
```

---

### `--D`

Latent width used for both heads of the DeepONet model, so that:

```text
Dp = Df = D
```

Example:

```bash
--D 400
```

Default:

```text
400
```

---

## Optimizer and Scheduler

Training uses:

```python
torch.optim.AdamW(...)
```

with weight decay:

```text
1e-9
```

The learning-rate scheduler is:

```python
ReduceLROnPlateau
```

configured with:
- `mode="min"`
- `factor=0.75`
- `patience=20`

The scheduler is stepped using the validation loss.

---

## Logging and Checkpointing

Every 50 epochs, the script saves:

### Loss history

```text
output_<out_name>/losses_latest.mat
```

This MATLAB file stores:
- `epoch`
- `train_total`
- `val_total`
- `time_50epoch_blocks`

### Model checkpoints

```text
output_<out_name>/deeponet_epoch_XXX.pt
```

where `XXX` is the epoch number.

Only rank 0 writes these files during distributed training.

---

## Typical Output Directory

A typical output directory has the form:

```text
output_Exp1/
├── normalisation_data.pt
├── losses_latest.mat
├── deeponet_epoch_050.pt
├── deeponet_epoch_100.pt
├── ...
└── saved_test_data/
    └── test_data.pt
```

The `normalisation_data.pt` file is required downstream. The internal `saved_test_data/` output may be produced by the training script, but it is not part of the main reported testing workflow.

---

## Running the Script Locally

For a local single-GPU run:

```bash
python main.py --num-samples 10000 --batch-size 32 --epochs 300 --out-name Exp1
```

For a smaller debug run:

```bash
python main.py --num-samples 1000 --batch-size 16 --epochs 20 --out-name debug_run
```

---

## Distributed Training with SLURM

The script supports distributed training through PyTorch DDP in a SLURM environment.

It uses the following SLURM environment variables:
- `SLURM_PROCID`
- `SLURM_LOCALID`
- `SLURM_NTASKS`
- `SLURM_LAUNCH_NODE_IPADDR` if available

The DDP setup:
- initializes the process group with backend `nccl`,
- assigns one GPU per task,
- wraps the model using `DistributedDataParallel`,
- uses `DistributedSampler` for both training and validation datasets.

This means that the number of SLURM tasks should match the number of GPUs requested.

---

## SLURM Batch Script

The following batch script is the version used for the reported runs, with account-specific settings removed. It is provided for reproducibility and may need to be adapted to a different cluster.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:3
##SBATCH --gres=gpu:ampere_a100:3
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# -------------------
# Args
# 1: num_samples (required)
# 2: epochs (optional, default 600)
# 3: D latent width (optional, default 400)
# -------------------
NUM_SAMPLES="${1:-}"
EPOCHS="${2:-600}"
D="${3:-400}"

if [[ -z "${NUM_SAMPLES}" ]]; then
  echo "Usage: sbatch RunDeep.sh <num_samples> [epochs=600] [D=400]"
  exit 1
fi

# Ensure D is a positive integer
if ! [[ "${D}" =~ ^[0-9]+$ ]] || [[ "${D}" -le 0 ]]; then
  echo "D must be a positive integer (got '${D}')."
  exit 1
fi

module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load GCC/13.2.0 OpenMPI/4.1.6 h5py/3.11.0
module load scikit-learn/1.4.0

# Output name
OUT_NAME="NS${NUM_SAMPLES}_D${D}"

# DDP: ntasks matches number of GPUs
srun --gpus-per-task=1 \
  python main.py \
    --ddp \
    --lr-schedule plateau \
    --out-name "${OUT_NAME}" \
    --num-samples "${NUM_SAMPLES}" \
    --epochs "${EPOCHS}" \
    --D "${D}" \
    --data-seed 529 \
    --weight-seed 1234
```

---

## Notes on the SLURM Script

This batch script assumes:
- one node,
- three tasks per node,
- three GPUs total,
- one GPU per task,
- a GPU partition,
- an environment where the specified modules are available.

You may need to adjust:
- GPU type,
- module names,
- memory allocation,
- CPU allocation,
- partition name,
- wall-clock time.

The commented line

```bash
##SBATCH --gres=gpu:ampere_a100:3
```

indicates that an alternative GPU type was also considered.

---

## Example SLURM Usage

To submit a run with:
- `num_samples = 10000`
- `epochs = 300`
- `D = 400`

use:

```bash
sbatch RunDeep.sh 10000 300 400
```

This will produce an output directory named:

```text
output_NS10000_D400/
```

---

## Expected Repository Layout

A suggested repository structure is:

```text
project_root/
├── main.py
├── RunDeep.sh
├── README.md
├── input_data_batch_seed_1.h5
├── output_data_batch_seed_1.h5
├── emulator_tools/
│   ├── models.py
│   ├── data_utils.py
│   └── train.py
└── MATLAB_files_for_emulator/
    ├── Nodes.mat
    └── MassMatrix.mat
```

---

## Practical Notes

- `Nodes.mat` and `MassMatrix.mat` must be available before training starts.
- `input_data_batch_seed_1.h5` and `output_data_batch_seed_1.h5` are required for training with the current `generate_data()` implementation.
- The code assumes that the HDF5 files contain datasets named `Input1`, `Input2`, `Input8`, `Output1`, and `Output2`.
- The script currently assumes 5 scalar inputs during training.
- The saved normalization statistics in `normalisation_data.pt` are required for downstream inference and evaluation.
- The internal test subset written by the training script is not the main route used for reported testing results.
- There is a current implementation mismatch between the number of branch channels loaded by `generate_data()` and the number expected by the model definition; this should be checked before reproducing the runs.
- Checkpoints are saved every 50 epochs.
- Validation uses:

```text
num_space_points_val = 500
```

- The training and validation routines are called with:

```text
bce_weight = 0.05
```

---

## Summary

This training pipeline provides a reproducible setup for training a DeepONet-based emulator of the resin-infusion model. It depends not only on the Python code, but also on MATLAB-generated geometry, mass-matrix, and HDF5 data files.

For reproducibility, the following should be archived together:
- the Python training script,
- `emulator_tools`,
- the SLURM launch script,
- `MATLAB_files_for_emulator/Nodes.mat`,
- `MATLAB_files_for_emulator/MassMatrix.mat`,
- the HDF5 data files used for training,
- the saved normalization statistics produced during training.