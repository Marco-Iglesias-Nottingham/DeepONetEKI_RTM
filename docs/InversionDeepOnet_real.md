# Inversion with DeepONet (real data)

# `EKI_real.py`

`EKI_real.py` performs ensemble Kalman inversion (EKI) using the trained DeepONet emulator and **real experimental data**.

It mirrors the synthetic inversion workflow, but replaces synthetic observations with measured data loaded from the `RealData/` folder. In addition to pressure data, it also incorporates inlet data into the observation vector.

The script uses:
- a trained DeepONet checkpoint,
- normalization statistics from training,
- emulator uncertainty statistics from testing,
- a prior ensemble in HDF5 format,
- real-data files from the experimental setup.

It produces:
- a posterior ensemble,
- a posterior scalar file,
- noisy emulator predictions,
- summary posterior statistics.

---

## Main Inputs

### 1. Training output

The script requires the normalization statistics saved during training:

```text
output_<exp_name>/normalisation_data.pt
```

This file is used to:
- normalize emulator branch inputs,
- normalize scalar inputs,
- de-normalize predicted pressure.

---

### 2. Testing / emulator uncertainty output

The script requires the testing output:

```text
output_<exp_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.pkl
```

This file provides the emulator uncertainty statistics used in inversion.

Depending on the chosen index file, the script loads one of:

- `for_real_sample_cov` and `for_real_mean_error`
- `idx100_sample_cov` and `idx100_mean_error`

For real-data inversion, the usual choice is the `_for_real` index set.

---

### 3. Trained model checkpoint

The trained emulator is loaded from:

```text
output_<exp_name>/deeponet_epoch_<epoch>.pt
```

If present, the script also supports index-tagged checkpoint names such as:

```text
deeponet_indices_for_real_epoch_<epoch>.pt
```

but it will fall back to the standard checkpoint filename if needed.

---

### 4. MATLAB geometry file

The script requires:

```text
MATLAB_files_for_emulator/Nodes.mat
```

This file provides the node coordinates used to construct the trunk input.

As in training and testing, the coordinates are scaled by `0.3`, so that the trunk network sees coordinates on `[0,1]^2`.

---

### 5. Real-data files

The script loads measured data from the `RealData/` folder.

For a chosen file number `file_no`, it expects:

```text
RealData/Real<file_no>.mat
RealData/Inlet<file_no>.mat
```

For example, if `file_no = 3`, the required files are:

```text
RealData/Real3.mat
RealData/Inlet3.mat
```

These files are used to build the real observation vector.

#### `Real<file_no>.mat`

This file is expected to contain:
- `ob_times`
- `press_full_corr`

The pressure data are interpolated onto the emulator time grid.

#### `Inlet<file_no>.mat`

This file is expected to contain:
- `inlet`

This inlet signal is also interpolated onto the selected time grid and appended to the observation vector.

---

### 6. Prior ensemble

The inversion starts from a prior ensemble stored in HDF5 format.

In the runs reported in the paper, the prior ensemble used was:

```text
Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

More generally, the path can be supplied through:

```text
--prior-file
```

The prior ensemble file is expected to contain:

- `/Input3`
- `/Input4`
- `/Input5`
- `/Input6`
- `/Input7`
- `/Input8`
- `/Input9`

These correspond to the inversion parameterization:
- level-set field,
- race-tracking geometry parameters,
- race-tracking permeability fields,
- scalar parameters,
- defect permeability field.

---

### 7. Index file

The script requires an index file specifying the pressure sensor locations used in the inversion.

For the real-data experiments, the paper uses:

```text
MATLAB_files_for_emulator/indices_for_real.mat
```

The variable name inside the file is assumed to be:

```text
ind
```

unless changed with:

```text
--index-var
```

The selected index file determines which covariance and mean-error arrays are taken from the testing `.pkl` file.

---

## What the Script Does

The script performs the following steps:

1. Load normalization statistics from training  
2. Load node coordinates and sensor indices  
3. Load real pressure and inlet data  
4. Interpolate real data onto the emulator time grid  
5. Load emulator uncertainty statistics from the testing output  
6. Load the trained DeepONet model  
7. Load a prior ensemble  
8. Build emulator inputs from the current ensemble  
9. Evaluate the emulator and inlet model  
10. Form a combined observation vector containing pressure and inlet data  
11. Build the total covariance from measurement error and emulator uncertainty  
12. Run iterative ensemble Kalman inversion  
13. Save posterior outputs and summary files  

---

## Observation Vector

Unlike `EKI_syn.py`, this real-data script augments the observation vector with **inlet data**.

The final observation vector contains:
- interpolated pressure observations at the selected sensors and times,
- interpolated inlet observations over the same selected times.

The emulator uncertainty covariance is applied only to the **pressure block**, while the inlet block is assigned zero emulator covariance.

Measurement-noise covariance is then added to both blocks to form the total covariance used for whitening.

---

## Inlet Model

The script includes a function:

```text
evaluate_inlet_ensemble_torch(...)
```

which evaluates the inlet model directly from the scalar parameters of each ensemble member.

This produces an inlet prediction for every ensemble member and selected time point. These inlet predictions are appended to the pressure predictions before inversion.

Thus the inversion uses a combined observation vector:
- pressure predictions from the DeepONet emulator,
- inlet predictions from the analytical inlet model.

---

## Time Handling

The emulator uses the fixed internal time grid:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
17, 19, 21, 23, 25, 27, 30, 35, 40, 45, 50, 55, 60,
65, 70, 80, 90, 100, 110
```

The real pressure and inlet data are interpolated onto the subset of these times up to a file-dependent final time:

- files `1` and `3`: final time `80`
- files `2` and `4`: final time `60`

This interpolated data is then used in the inversion.

---

## Inversion Method

The script performs iterative ensemble Kalman inversion.

At each iteration:
- the current ensemble is mapped to predicted pressure fields,
- inlet predictions are evaluated from the scalar parameters,
- the full predicted observation vector is built,
- the residual is whitened using the total covariance,
- the ensemble is updated using Kalman-type formulas.

The total covariance includes:
- measurement-error covariance,
- emulator covariance on the pressure block.

The emulator mean error is subtracted from the pressure observations before inversion.

---

## Parameter Transformations

Some scalar parameters are transformed before inversion using bounded log-ratio transforms.

The script applies:
- `TransformAll(...)` before inversion,
- `InvTransformAll(...)` after each update.

This keeps updated scalar parameters inside the prescribed admissible ranges.

---

## Emulator Forward Model

The emulator forward map is evaluated using the trained `DeepONetPressureFront` model.

For each ensemble member:
- permeability and porosity fields are built from the parameterized representation,
- these are normalized with the saved training statistics,
- trunk coordinates are built from the selected sensor locations and normalized times,
- the model predicts pressure and front,
- pressure is de-normalized and masked using the thresholded predicted front.

This is done in batches using:

```text
batched_forward(...)
```

to control GPU memory usage.

---

## Outputs Created

The script saves several outputs.

### 1. Posterior ensemble

The main posterior ensemble is saved as:

```text
output_<exp_name>/posterior_real<file_no>.h5
```

This file contains:

- `/Input3`
- `/Input4`
- `/Input5`
- `/Input6`
- `/Input7`
- `/Input8`
- `/Input9`

that is, the posterior ensemble in the same parameterized form as the prior ensemble.

---

### 2. Posterior scalar file

A companion file is saved as:

```text
output_<exp_name>/posterior_real<file_no>_scalar.h5
```

This contains:

- `/Input8`

for convenience.

---

### 3. Posterior prediction file

A prediction file is saved as:

```text
output_<exp_name>/posterior_real<file_no>_pred.h5
```

This file contains:

- `Predictions`
- `real_data`
- `real_time`

Here:
- `Predictions` contains emulator predictions with correlated noise added,
- `real_data` contains the augmented real observation vector,
- `real_time` stores the selected time points used in the inversion.

---

### 4. Summary statistics

The script saves a posterior summary file:

```text
output_<exp_name>/summary_vars_real<file_no>.npz
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

### 5. Convergence information

During the EKI iterations, the script writes:

```text
converged.mat
```

This stores:
- `Misfit_ave`
- `t`
- `iter`
- `alpha`
- `alpha_0`

and can be used to monitor convergence.

---

## Command-Line Arguments

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

### `--index-file`

Path to the `.mat` file defining the pressure sensor indices.

For the real-data experiments in the paper, this is:

```bash
--index-file MATLAB_files_for_emulator/indices_for_real.mat
```

---

### `--index-var`

Variable name inside the index `.mat` file.

Default:

```text
ind
```

---

### `--prior-file`

Path to the prior ensemble HDF5 file.

In the paper runs, this was:

```bash
--prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

---

### `--posterior-out`

Optional explicit output path for the posterior ensemble file.

If omitted, the script generates the default filename automatically.

---

### `--file-no`

Number of the real-data case to use.

Examples:

```bash
--file-no 1
--file-no 2
--file-no 3
--file-no 4
```

This selects:
- `RealData/Real<file_no>.mat`
- `RealData/Inlet<file_no>.mat`

These correspond to the **four real-data cases reported in the paper**.

---

### `--seed`

Random seed for reproducibility.

Default:

```text
42
```

In the SLURM runs used for the paper, the seed was set to:

```text
13278
```

---

### `--deterministic`

Enable more deterministic PyTorch behavior. This may be slower.

This flag was used in the paper runs.

---

### `--D`

Latent width of the DeepONet model.

This must match the model used during training and testing.

Default:

```text
400
```

---

## Example Command

A typical real-data inversion run is:

```bash
python EKI_real.py \
  --exp-name NS40000_D400 \
  --epoch 400 \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --file-no 3 \
  --seed 13278 \
  --deterministic \
  --D 400
```

---

## Reproducibility Notes

For reproducibility, the real-data inversions in the paper used:

- experiment name of the form:

```text
NS<num_samples>_D<D>
```

- a single GPU run,
- `indices_for_real.mat`,
- prior ensemble:

```text
Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

- seed:

```text
13278
```

- deterministic mode enabled,
- real-data cases:

```text
file_no = 1, 2, 3, 4
```

Thus, the **four cases from the paper** are obtained by running `EKI_real.py` four times with the same trained model, prior ensemble, and index set, but with:

- `--file-no 1`
- `--file-no 2`
- `--file-no 3`
- `--file-no 4`

The normalization statistics, trained checkpoint, and testing/UQ `.pkl` file must all come from the **same training/testing run**.

The node ordering must be consistent across:
- `Nodes.mat`,
- the index file,
- the emulator outputs,
- the prior ensemble fields,
- the real-data sensor interpretation.

---

## SLURM Launch Settings Used for the Paper Runs

The four real-data cases reported in the paper were launched with a SLURM script that ran `EKI_real.py` once for each `file_no = 1,2,3,4`.

The relevant settings were:
- **1 node**
- **1 task**
- **1 GPU**
- **20 CPUs per task**
- **24 hour wall-clock limit**
- module-based PyTorch environment
- fixed seed `13278`
- deterministic mode enabled
- `indices_for_real.mat` used in all four runs
- common prior ensemble:
  
```text
Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5
```

A simplified version of the launch script is:

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

NUM_SAMPLES="${1:-}"
EPOCH="${2:-400}"
D="${3:-400}"

module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load GCC/13.2.0 OpenMPI/4.1.6 h5py/3.11.0
module load scikit-learn/1.4.0

EXP_NAME="NS${NUM_SAMPLES}_D${D}"

srun --gpus-per-task=1 python EKI_real.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}" \
  --file-no 1

srun --gpus-per-task=1 python EKI_real.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}" \
  --file-no 2

srun --gpus-per-task=1 python EKI_real.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}" \
  --file-no 3

srun --gpus-per-task=1 python EKI_real.py \
  --exp-name "${EXP_NAME}" \
  --epoch "${EPOCH}" \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --seed 13278 \
  --deterministic \
  --D "${D}" \
  --file-no 4
```

---

## Required Folder Structure

A typical folder layout is:

```text
project_root/
├── EKI_real.py
├── emulator_tools/
│   ├── models.py
│   └── data_utils.py
├── MATLAB_files_for_emulator/
│   ├── Nodes.mat
│   ├── indices_for_real.mat
│   └── indices_100.mat
├── RealData/
│   ├── Real1.mat
│   ├── Real2.mat
│   ├── Real3.mat
│   ├── Real4.mat
│   ├── Inlet1.mat
│   ├── Inlet2.mat
│   ├── Inlet3.mat
│   └── Inlet4.mat
├── Full_EKI_outputs/
│   └── prior_ensemble_100_NEn5000_seed91882.h5
└── output_<exp_name>/
    ├── normalisation_data.pt
    ├── deeponet_epoch_<epoch>.pt
    └── test_outputs_epoch/
        └── errors_and_samples_epoch_<epoch>.pkl
```

---

## Summary

`EKI_real.py` performs real-data inversion with the DeepONet emulator.

It combines:
- training outputs,
- testing-derived emulator uncertainty,
- measured pressure and inlet data,
- and a prior ensemble,

to produce:
- a posterior ensemble,
- posterior scalar files,
- noisy posterior predictions,
- and posterior summary statistics.

The four real-data cases reported in the paper are obtained by running the script with:

- `--file-no 1`
- `--file-no 2`
- `--file-no 3`
- `--file-no 4`

using the same:
- trained checkpoint,
- normalization statistics,
- testing/UQ output,
- prior ensemble,
- and `indices_for_real.mat`.

For reproducibility, archive together:
- `EKI_real.py`
- the trained checkpoint
- `normalisation_data.pt`
- the testing/UQ `.pkl` file
- `Nodes.mat`
- `indices_for_real.mat`
- the `RealData/` files
- the prior ensemble HDF5 file
- the SLURM launch script or equivalent run settings.