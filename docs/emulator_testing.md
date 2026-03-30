# DeepONet Testing and Evaluation Script

This script performs **offline evaluation** of a trained `DeepONetPressureFront` model for the resin-infusion emulator.

It:
- loads a trained model checkpoint,
- loads test data generated from the MATLAB pipeline,
- applies the saved normalization statistics from training,
- performs prediction in trunk-chunks to reduce GPU memory usage,
- computes pressure and front errors using the FEM mass matrix,
- computes mean and covariance information on sensor-index subsets,
- saves all results in formats suitable for later post-processing in MATLAB or Python.

---

## Main Script

The testing script is:

```text
test_model.py
```

It depends on:
- a trained model checkpoint,
- the normalization statistics saved during training,
- MATLAB-generated geometry and index files,
- HDF5 test data files,
- the local Python package `emulator_tools`.

---

## Overview

The testing workflow is:

1. Load a trained model checkpoint  
2. Load normalization statistics from training  
3. Load test data from HDF5 files  
4. Apply the same normalization used during training  
5. Run predictions in chunks to reduce GPU memory usage  
6. De-normalize pressure predictions  
7. Compute pressure and front errors using the FEM mass matrix  
8. Compute mean and covariance statistics on selected sensor-index sets  
9. Save outputs for later analysis and inversion  

---

## Required Files

### Trained model checkpoint

The script expects a model checkpoint in:

```text
output_<out_name>/deeponet_epoch_<epoch>.pt
```

For example:

```text
output_NS10000_D400/deeponet_epoch_500.pt
```

---

### Normalization statistics

The script also requires:

```text
output_<out_name>/normalisation_data.pt
```

This file is produced during training and is required to:
- normalize the test inputs consistently with training,
- de-normalize the pressure predictions.

---

## Dependencies on MATLAB-Generated Files

The script depends directly on several files produced by the MATLAB workflow.

These are expected in:

```text
MATLAB_files_for_emulator/
```

Specifically:

- `Nodes.mat`
- `MassMatrix.mat`
- `indices_for_real.mat`
- `indices_100.mat`

### `Nodes.mat`

Used in `generate_data_for_testing()` to build trunk inputs from the spatial coordinates.

### `MassMatrix.mat`

Used to compute FEM-based error norms for:
- pressure,
- filling factor.

### `indices_for_real.mat` and `indices_100.mat`

Used to define sensor-index subsets on which mean and covariance error statistics are computed.

These correspond to:
- the `_for_real` sensor configuration,
- the `_100` sensor configuration.

---

## Dependencies on HDF5 Test Data

The function `generate_data_for_testing()` currently loads test data from the hardcoded files:

```text
input_data_batch_seed_2.h5
output_data_batch_seed_2.h5
```

These files must contain:

### Input file
- `/Input1`: permeability field  
- `/Input2`: porosity field  
- `/Input8`: scalar parameters  

### Output file
- `/Output1`: pressure  
- `/Output2`: filling factors  

---

## Running the Script

Example:

```bash
python test_model.py --out-name NS10000_D400 --epoch 500 --D 400
```

This loads:

- `output_NS10000_D400/deeponet_epoch_500.pt`
- `output_NS10000_D400/normalisation_data.pt`

and writes results to:

```text
output_NS10000_D400/test_outputs_epoch/
```

---

## Command-Line Arguments

### `--out-name`

Name suffix of the output directory from training.

Example:

```bash
--out-name NS10000_D400
```

This corresponds to:

```text
output_NS10000_D400/
```

---

### `--epoch`

Checkpoint epoch to load.

Example:

```bash
--epoch 500
```

This loads:

```text
deeponet_epoch_500.pt
```

---

### `--D`

Latent width used in the model.

Example:

```bash
--D 400
```

This must match the value used during training.

---

## Prediction Strategy

Predictions are performed using chunked trunk evaluation through:

```python
predict_chunked(...)
```

This:
- flattens the trunk inputs,
- evaluates the model in manageable chunks,
- avoids GPU memory overflow,
- concatenates the prediction chunks afterward.

The script uses mixed precision with `bfloat16` during inference for efficiency.

---

## Normalization and De-normalization

Inputs are normalized using the saved training statistics via:

```python
apply_saved_normalization(...)
```

Pressure outputs are then de-normalized using the saved pressure mean and standard deviation:

```python
p_pred_un = p_pred * p_std + p_mean
```

The pressure targets are de-normalized in the same way before error computation.

---

## Coordinate Scaling and Mass Matrix

The trunk coordinates are normalized to `[0,1]^2` before being passed to the network.

However, the error metrics are computed using the finite-element mass matrix on the original physical domain. This is consistent: the coordinate normalization is only a preprocessing step for the neural network input, while the FEM mass matrix defines the physical integration measure used in evaluation.

Thus:
- the network sees normalized coordinates,
- the reported error norms remain tied to the original FEM discretization.

---

## Error Metrics

All spatial error metrics are computed using:

```python
M = loadmat("MATLAB_files_for_emulator/MassMatrix.mat")["MassMatrix"]
```

The code converts this matrix to a dense PyTorch tensor and uses it in quadratic forms.


## Index-Set Mean and Covariance Statistics

In addition to global error metrics, the script computes error statistics on two predefined index sets:

- `for_real`
- `100`

These index sets are loaded from:

- `MATLAB_files_for_emulator/indices_for_real.mat`
- `MATLAB_files_for_emulator/indices_100.mat`

For each index set, the script extracts the pressure prediction error restricted to those sensor locations, then computes:

- the **mean error vector**,
- the **sample covariance matrix**.

These quantities are saved because they are used later in the inversion workflow.

---

## Output Folder

The testing script writes outputs to:

```text
output_<out_name>/test_outputs_epoch/
```

For example:

```text
output_NS10000_D400/test_outputs_epoch/
```

---

## Files Created by the Testing Script

For a given epoch, the script creates:

### MATLAB output

```text
output_<out_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.mat
```

### Python pickle output

```text
output_<out_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.pkl
```

These two files contain the same core information in MATLAB and Python-friendly formats.

---

## Saved Quantities

The output files include:

### Global metrics

- `rel_l2_mean`
- `rel_l2_std`
- `rel_l2_front_mean`
- `rel_l2_front_std`
- `rel_l1_front_mean`
- `rel_l1_front_std`

### Saved prediction samples

A small number of representative samples are stored for inspection and plotting:

- `p_preds_samples`
- `p_trues_samples`
- `f_preds_samples`
- `f_trues_samples`
- `branch_samples`
- `scalar_branch_samples`

### Mean and covariance information used in inversion

The following arrays are saved:

- `for_real_mean_error`
- `for_real_sample_cov`
- `idx100_mean_error`
- `idx100_sample_cov`

These correspond to:
- the mean pressure error vector on the `_for_real` sensor set,
- the sample covariance matrix on the `_for_real` sensor set,
- the mean pressure error vector on the `_100` sensor set,
- the sample covariance matrix on the `_100` sensor set.

These saved quantities are the testing outputs that are later used in the inversion workflow.

---

## Role of the Saved Mean and Covariance Files in Inversion

The files

```text
errors_and_samples_epoch_<epoch>.mat
errors_and_samples_epoch_<epoch>.pkl
```

stored in

```text
output_<out_name>/test_outputs_epoch/
```

contain the mean and covariance information needed downstream for inversion.

In particular, the arrays

- `for_real_mean_error`
- `for_real_sample_cov`
- `idx100_mean_error`
- `idx100_sample_cov`

provide the statistics associated with the two sensor configurations and can be loaded later in MATLAB or Python for inversion-related calculations.

---

## Consistency Requirements

For the testing and inversion workflow to be valid, the following must use the same node ordering:

- `Nodes.mat`
- `MassMatrix.mat`
- `indices_for_real.mat`
- `indices_100.mat`
- the HDF5 output fields `Output1` and `Output2`

If these are not consistent, the error norms and index-set statistics will not correspond to the intended spatial locations.

---

## SLURM Batch Script for Testing

The following SLURM batch script was used to run the testing pipeline. Account-specific settings have been removed.

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
##SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=3850
#SBATCH --gres=gpu:lovelace_l40:1
##SBATCH --gres=gpu:ampere_a100:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Modules ---
module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load GCC/13.2.0 OpenMPI/4.1.6 h5py/3.11.0
module load scikit-learn/1.4.0

# --- Args ---
# Usage: sbatch submit_test.sh <num_samples> <epoch> <D>
NUM_SAMPLES="${1:-}"
EPOCH="${2:-}"
D="${3:-}"

if [[ -z "$NUM_SAMPLES" || -z "$EPOCH" || -z "$D" ]]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch submit_test.sh <num_samples> <epoch> <D>"
  echo "Example: sbatch submit_test.sh 10000 500 400"
  exit 1
fi

# Validate numerics
if ! [[ "$NUM_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "num_samples must be an integer (got: '$NUM_SAMPLES')"
  exit 1
fi
if ! [[ "$EPOCH" =~ ^[0-9]+$ ]]; then
  echo "epoch must be an integer (got: '$EPOCH')"
  exit 1
fi
if ! [[ "$D" =~ ^[0-9]+$ ]]; then
  echo "D must be an integer (got: '$D')"
  exit 1
fi

# Output name (must match training naming convention)
OUT_NAME="NS${NUM_SAMPLES}_D${D}"

echo "Running test_model.py with:"
echo "  out-name = ${OUT_NAME}"
echo "  epoch    = ${EPOCH}"
echo "  D        = ${D}"

# --- Run ---
srun --gpus-per-task=1 \
  python test_model.py \
    --out-name "${OUT_NAME}" \
    --epoch "${EPOCH}" \
    --D "${D}"
```

---

## Notes on the SLURM Script

This script assumes:
- one node,
- one task,
- one GPU,
- a GPU-enabled PyTorch installation,
- an environment where the listed modules are available.

You may need to adapt:
- module names,
- GPU type,
- CPU and memory requests,
- partition name,
- wall-clock limit.

The output naming convention in the script is:

```text
OUT_NAME="NS${NUM_SAMPLES}_D${D}"
```

so it must match the naming convention used during training.

---

## Example SLURM Usage

```bash
sbatch submit_test.sh 10000 500 400
```

This evaluates:
- the model trained with `10000` samples,
- the checkpoint at epoch `500`,
- latent width `D = 400`.

It will use the trained model and normalization files in:

```text
output_NS10000_D400/
```

and write test results to:

```text
output_NS10000_D400/test_outputs_epoch/
```

---

## Expected Folder Structure

A typical layout is:

```text
project_root/
├── test_model.py
├── submit_test.sh
├── input_data_batch_seed_2.h5
├── output_data_batch_seed_2.h5
├── emulator_tools/
│   ├── models.py
│   ├── data_utils.py
│   └── ...
├── MATLAB_files_for_emulator/
│   ├── Nodes.mat
│   ├── MassMatrix.mat
│   ├── indices_for_real.mat
│   └── indices_100.mat
└── output_NS10000_D400/
    ├── deeponet_epoch_500.pt
    ├── normalisation_data.pt
    └── test_outputs_epoch/
        ├── errors_and_samples_epoch_500.mat
        └── errors_and_samples_epoch_500.pkl
```

---

## Summary

This script provides the offline testing pipeline for the DeepONet resin-infusion emulator. It:
- applies the same normalization used during training,
- evaluates a saved model checkpoint on test data,
- computes FEM-based pressure and front errors,
- computes mean and covariance statistics on sensor-index subsets,
- saves MATLAB and Python output files for later analysis.

For reproducibility, the following should be archived together:
- `test_model.py`
- the SLURM submission script,
- the trained checkpoint,
- `normalisation_data.pt`,
- `Nodes.mat`
- `MassMatrix.mat`
- `indices_for_real.mat`
- `indices_100.mat`
- the HDF5 test files
- the output folder `test_outputs_epoch/` containing the saved error statistics used later in inversion.