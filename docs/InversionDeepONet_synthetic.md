# Inversion with DeepONet (synthetic data)

`EKI_syn.py` generates a posterior ensemble for synthetic experiments using the trained DeepONet emulator.

The inversion workflow uses:
- a trained DeepONet model,
- normalization statistics saved during training,
- emulator error mean and covariance estimated during testing,
- synthetic data,
- a prior ensemble stored in HDF5 format.

`EKI_syn.py` performs an iterative EKI update until convergence.

The script saves:
- the posterior ensemble,
- summary statistics of the posterior fields,
- emulator predictions with observation +emulator error.

---

## Main Inputs

The script depends on several files produced in earlier stages of the workflow.

### 1. Training output

From the training stage, it requires:

```text
output_<exp_name>/normalisation_data.pt
```

This file contains the normalization and standardization statistics used to:
- normalize branch inputs,
- normalize scalar inputs,
- de-normalize predicted pressure.

---

### 2. Testing output

From the testing stage, it requires:

```text
output_<exp_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.pkl
```

This file provides the emulator error statistics used in inversion.

In particular, the script uses the filtered covariance and mean-error quantities produced by `test_model.py`:

- `for_real_sample_cov`
- `for_real_mean_error`
- `idx100_sample_cov`
- `idx100_mean_error`

The choice between these depends on the index file supplied to the script.

---

### 3. Trained model checkpoint

The trained DeepONet model is loaded from:

```text
output_<exp_name>/deeponet_epoch_<epoch>.pt
```

or, if fine-tuned models are used, from:

```text
output_<exp_name>_finetune/
```

The script supports both standard and fine-tuned checkpoint naming conventions.

---

### 4. MATLAB geometry file

The script requires:

```text
MATLAB_files_for_emulator/Nodes.mat
```

This file provides the node coordinates used to build the trunk input for the emulator.

As in training and testing, the coordinates are scaled by `0.3` so that the trunk network sees coordinates on `[0,1]^2`.

---

### 5. Synthetic data file

For synthetic inversion, the script loads a MATLAB `.mat` file containing the synthetic observations.

Examples include:

```text
MATLAB_files_for_emulator/syndata_100.mat
MATLAB_files_for_emulator/syndata_for_real.mat
```

The synthetic-data file is expected to contain:
- `syn_data`: syntethic data (with Gaussian noise)
- `noise_free_data`: syntethic data (before Gaussian noise is added)
- `Error_std`: standard deviation of the noise added
- `mask`: true log permeability field
- `mask2`: true porosity field


---

### 6. Prior ensemble

The inversion starts from a prior ensemble stored in HDF5 format.

A typical input is:

```text
Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

The prior ensemble file is expected to contain:

- `Input3`: $L(x)$  
- `Input4`: $\log K_T(x)$  
- `Input5`: $\log K_B(x)$  
- `Input6`: $\xi_T$  
- `Input7`: $\xi_B$  
- `Input9`: $\log K_{\mathrm{def}}(x)$  

Scalar parameters are stored in `Input8`:

- `Input8(1,:)`: $K_{\text{nom}}$  
- `Input8(2,:)`: Mean of $\log K_{\text{def}}$; this is not actually used in the inversion, since the inversion is performed directly on $\log K_{\text{def}}(x)$ stored in `Input9`  
- `Input8(3,:)`: $\phi_{\text{nom}}$  
- `Input8(4,:)`: $\phi_{\text{def}}$  
- `Input8(5,:)`: $\phi_T$  
- `Input8(6,:)`: $\phi_B$  
- `Input8(7,:)`: $\mu$  
- `Input8(8,:)`: $P_I$  
- `Input8(9,:)`: $\gamma$  
- `Input8(10,:)`: $\beta$  
- `Input8(11,:)`: $\chi$  

---

### 7. Index file

The script also requires an index file specifying the observation/sensor locations used in the inversion.

This is passed through:

```text
--index-file
```

Typical examples are:

```text
MATLAB_files_for_emulator/indices_for_real.mat
MATLAB_files_for_emulator/indices_100.mat
```

The variable name inside the `.mat` file is assumed to be:

```text
ind
```

unless a different variable name is passed with `--index-var`.

The index file determines which covariance and mean-error quantities are read from the testing `.pkl` file:
- `indices_for_real.mat`  → `for_real_*`
- `indices_100.mat`       → `idx100_*`

---

## What the Script Does

The script performs the following steps:

1. Load normalization statistics from training  
2. Load node coordinates and observation indices  
3. Load synthetic observation data  
4. Load emulator uncertainty statistics from the testing output  
5. Load the trained DeepONet model  
6. Load a prior ensemble of inversion parameters  
7. Build emulator inputs from the prior ensemble  
8. Compute emulator predictions at the selected observation indices  
9. Form the total observation covariance as measurement noise plus emulator covariance  
10. Run iterative ensemble Kalman inversion  
11. Save the posterior ensemble and summary outputs  

---



## Parameter Transformations

Some scalar parameters are transformed before inversion using bounded log-ratio transforms.

The script defines limits for each scalar parameter and applies:
- `TransformAll(...)` before inversion,
- `InvTransformAll(...)` after each update.

This ensures that updated scalar parameters remain within the prescribed admissible ranges.

---

## Emulator Forward Model

The emulator forward map is evaluated using the trained `DeepONetPressureFront` model.

For each ensemble member:
- permeability and porosity fields are built from the parameterized representation,
- inputs are normalized with the saved training statistics,
- trunk coordinates are built from the selected sensor locations and normalized times,
- the model predicts pressure and filling factor,
- predicted pressure is de-normalized and masked using the thresholded front.

This is done in batches using:

```text
batched_forward(...)
```

to limit GPU memory usage.

---

## Outputs Created

The script saves several outputs.

### 1. Posterior ensemble

The main posterior ensemble is saved as an HDF5 file, typically in:

```text
output_<exp_name>/posterior_ensemble_<tag>_epoch_<epoch>.h5
```

or, for fine-tuned runs, with `_fine` in the filename.

This file contains:

- `/Input3`
- `/Input4`
- `/Input5`
- `/Input6`
- `/Input7`
- `/Input8`
- `/Input9`

that is, the updated posterior ensemble in the same parameterized form as the prior ensemble.

---

### 2. Posterior scalar file

A second HDF5 file is also written containing:

```text
/Input8
```

for convenience.

---

### 3. Summary statistics

The script saves summary posterior statistics as an `.npz` file, typically named:

```text
summary_vars_<tag>_epoch_<epoch>.npz
```

This file contains:

- `perm_std`
- `perm_mean`
- `poro_std`
- `poro_mean`
- `mean_LS`
- `mean_RT`

These summarize the posterior ensemble in terms of:
- mean and standard deviation of permeability,
- mean and standard deviation of porosity,
- mean level-set indicator,
- mean race-tracking indicator.

---

### 4. Noisy predicted observations

The script also saves emulator predictions with correlated noise added, in a companion HDF5 file with:

```text
Predictions
```

This uses the same covariance model employed in the inversion.

---

### 5. Convergence information

During the EKI iterations, the script writes:

```text
converged.mat
```

This file stores:
- `Misfit_ave`
- `t`
- `iter`
- `alpha`
- `alpha_0`

and is used to monitor inversion progress.

---

## Command-Line Arguments

The main command-line options are:

### `--exp-name`

Base experiment name used to locate:

```text
output_<exp_name>/
```

Example:

```bash
--exp-name NS40000_D400
```

---

### `--epoch`

Epoch number used to load:
- the trained checkpoint,
- the testing/UQ `.pkl` file.

Example:

```bash
--epoch 450
```

---

### `--finetuned`

If provided, the script uses:

```text
output_<exp_name>_finetune/
```

and searches for fine-tuned checkpoint filenames.

---

### `--index-file`

Path to the `.mat` file defining the observation indices.

Example:

```bash
--index-file MATLAB_files_for_emulator/indices_for_real.mat
```

or

```bash
--index-file MATLAB_files_for_emulator/indices_100.mat
```

---

### `--index-var`

Variable name inside the index `.mat` file.

Default:

```text
ind
```

---

### `--syn-file`

Synthetic-data `.mat` file.

Examples:

```bash
--syn-file MATLAB_files_for_emulator/syndata_for_real.mat
```

```bash
--syn-file MATLAB_files_for_emulator/syndata_100.mat
```

---

### `--prior-file`

Path to the prior ensemble HDF5 file.

Example:

```bash
--prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

---

### `--posterior-out`

Optional explicit output path for the posterior ensemble file.

If omitted, the script creates a default name automatically.

---

### `--seed`

Random seed for reproducibility.

Default:

```text
42
```

---

### `--deterministic`

Enable more deterministic PyTorch behavior. This may be slower.

---

### `--D`

Latent width of the DeepONet model.

This must match the model used during training and testing.

Default:

```text
400
```

---

## Reproducibility Notes

For reproducibility, the inversion runs were launched on a single GPU with a SLURM submission script. The important aspects are:

- **1 node**
- **1 task**
- **1 GPU**
- **20 CPUs per task**
- **24 hour wall-clock limit**
- PyTorch environment loaded through cluster modules
- deterministic mode enabled with:
  
```bash
--deterministic
```

- inversion seed fixed to:

```text
13278
```

The experiment name was constructed as:

```text
NS<num_samples>_D<D>
```

so that it matches the folder naming convention used during training and testing:

```text
output_NS<num_samples>_D<D>/
```

The inversion was run for both synthetic sensor configurations:

- `indices_100.mat` with `syndata_100.mat`
- `indices_for_real.mat` with `syndata_for_real.mat`

using the same prior ensemble:

```text
Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

---

## Example Commands

A typical synthetic inversion run for the `100`-sensor case is:

```bash
python EKI_syn.py \
  --exp-name NS40000_D400 \
  --epoch 400 \
  --index-file MATLAB_files_for_emulator/indices_100.mat \
  --syn-file MATLAB_files_for_emulator/syndata_100.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D 400
```

A corresponding run for the `_for_real` sensor configuration is:

```bash
python EKI_syn.py \
  --exp-name NS40000_D400 \
  --epoch 400 \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --syn-file MATLAB_files_for_emulator/syndata_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D 400
```

---

## SLURM Batch Script Used for Synthetic Inversion

The following SLURM script captures the relevant launch settings used for reproducibility, with account-specific information removed.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:1
##SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Usage: sbatch RunDeep.sh <num_samples> [epoch] [D]
NUM_SAMPLES="${1:-}"
EPOCH="${2:-400}"
D="${3:-400}"

if [[ -z "$NUM_SAMPLES" ]]; then
  echo "Usage: sbatch RunDeep.sh <num_samples> [epoch] [D]"
  exit 1
fi

module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load GCC/13.2.0 OpenMPI/4.1.6 h5py/3.11.0
module load scikit-learn/1.4.0

EXP_NAME="NS${NUM_SAMPLES}_D${D}"

# 100-sensor synthetic case
srun --gpus-per-task=1 python EKI_syn.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_100.mat \
  --syn-file MATLAB_files_for_emulator/syndata_100.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}"

# "for_real" synthetic case
srun --gpus-per-task=1 python EKI_syn.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --syn-file MATLAB_files_for_emulator/syndata_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}"
```

---

## Required Folder Structure

A typical folder layout is:

```text
project_root/
├── InversionDeepOnet.md
├── EKI_syn.py
├── emulator_tools/
│   ├── models.py
│   └── data_utils.py
├── MATLAB_files_for_emulator/
│   ├── Nodes.mat
│   ├── indices_for_real.mat
│   ├── indices_100.mat
│   ├── syndata_for_real.mat
│   └── syndata_100.mat
├── Full_EKI_outputs/
│   └── prior_ensemble_100_NEn5000_seed91882.h5
└── output_<exp_name>/
    ├── normalisation_data.pt
    ├── deeponet_epoch_<epoch>.pt
    └── test_outputs_epoch/
        └── errors_and_samples_epoch_<epoch>.pkl
```

---

## Notes

- The script assumes that the uncertainty statistics in the testing `.pkl` file were generated using the same model checkpoint and sensor/index configuration.
- The value of `D` must match the latent width used to train the model.
- The normalization file must come from the same training run as the loaded checkpoint.
- The node ordering must be consistent across:
  - `Nodes.mat`
  - the prior ensemble fields
  - the synthetic data
  - the selected observation indices
  - the emulator outputs used in testing.

---

## Summary

`EKI_syn.py` performs synthetic-data inversion with the DeepONet emulator using an ensemble Kalman inversion procedure.

It combines:
- training outputs,
- testing-derived emulator uncertainty,
- MATLAB-generated synthetic data,
- and a prior ensemble,

to produce:
- a posterior ensemble,
- posterior summary statistics,
- and noisy predicted observations for downstream analysis.

For reproducibility, archive together:
- `EKI_syn.py`
- the trained model checkpoint
- `normalisation_data.pt`
- the testing/UQ `.pkl` file
- `Nodes.mat`
- the relevant index and synthetic-data `.mat` files
- the prior ensemble HDF5 file
- the SLURM launch script or equivalent run settings.