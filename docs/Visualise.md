# Visualisation

The folder `Visualise/` contains plotting utilities for inspecting emulator predictions and saved testing outputs.

In particular,

```text
Visualise/Plot_emulator.py
```

loads saved results from the testing stage and produces figures for:
- pressure predictions versus targets,
- filling-factor predictions versus targets,
- absolute error fields,
- input permeability and porosity fields.

The script is intended for post-processing and figure generation rather than training or testing.

---

## Main Dependencies

`Plot_emulator.py` depends on the following Python packages:

- `numpy`
- `torch`
- `matplotlib`
- `scipy`
- `pickle`

In particular, it uses:
- `matplotlib.pyplot`
- `matplotlib.tri`
- `scipy.io.loadmat`

If LaTeX rendering is enabled in Matplotlib, a working LaTeX installation is also required.

---

## Files Required

The script depends on files produced earlier in the workflow.

### 1. Testing output file

The main inputs are the saved outputs from `test_model.py`, for example:

```text
output_<out_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.pkl
```

or alternatively:

```text
output_<out_name>/test_outputs_epoch/errors_and_samples_epoch_<epoch>.mat
```

The current example in the script loads the `.pkl` file.

These files must contain arrays such as:
- `p_preds_samples`
- `p_trues_samples`
- `f_preds_samples`
- `f_trues_samples`
- `branch_samples`
- `scalar_branch_samples`

---

### Default provided results (`output_data/`)

The repository includes a precomputed results folder:

```text
output_data/
```

This corresponds to the **best-performing trained model**, with:

- **40,000 training samples**
- **latent width** \(D = 400\)
- **450 training epochs**

The files in this folder can be used directly with `Plot_emulator.py` without re-running training or testing.

In particular, the script currently uses:

```text
output_data/errors_and_samples_epoch_450.pkl
output_data/normalisation_data.pt
```

---

### 2. Normalization statistics

The script also loads:

```text
output_<out_name>/normalisation_data.pt
```

(or from `output_data/` for the provided run)

This is needed to de-normalize:
- branch fields,
- scalar parameters,
- pressure outputs if required.

---

### 3. MATLAB geometry file

The script requires:

```text
MATLAB_files_for_emulator/Nodes.mat
```

This file provides the node coordinates used for unstructured triangulation and plotting of:
- pressure,
- filling factor,
- error fields.

---

## Output Files Created

The script writes figures into a local `Figures/` folder.

Typical outputs include:

```text
Figures/pressure_emulator.png
Figures/front_emulator.png
Figures/poro_perm.png
```

So the directory

```text
Figures/
```

must either exist already or be creatable from the working directory.

---

## What the Script Produces

### Pressure plots

The script creates three-row unstructured plots showing:
- target pressure,
- predicted pressure,
- absolute pressure error,

at selected time instances.

---

### Filling-factor plots

The same style of plot is used for the filling factor:
- target front / filling factor,
- predicted front / filling factor,
- absolute error.

---

### Input field plots

The script also reconstructs and plots:
- `\log K`
- `\phi`

from the normalized saved branch inputs.

---

## Important Path Assumptions

The current script contains hardcoded example paths such as:

```text
output_data/errors_and_samples_epoch_450.pkl
output_data/normalisation_data.pt
../MATLAB_files_for_emulator/Nodes.mat
Figures/pressure_emulator.png
```

If you are using a different experiment, these should be updated accordingly, for example:

```text
output_NS10000_D400/test_outputs_epoch/errors_and_samples_epoch_500.pkl
output_NS10000_D400/normalisation_data.pt
MATLAB_files_for_emulator/Nodes.mat
```

---

## Notes on Data Shapes

The script assumes that saved outputs from the testing pipeline can be reshaped into:

```text
(N_samples, N_time, N_space)
```

for both pressure and filling-factor fields.

In the example currently included in the file, the dimensions are:

```text
B = 625
T = 34
S = 2973
```

These must match the saved testing outputs.

---

## Suggested Folder Structure

```text
project_root/
├── Visualise/
│   └── Plot_emulator.py
├── Figures/
├── MATLAB_files_for_emulator/
│   └── Nodes.mat
├── output_data/   ← provided best model results
└── output_<out_name>/   ← optional other runs
    ├── normalisation_data.pt
    └── test_outputs_epoch/
        └── errors_and_samples_epoch_<epoch>.pkl
```

---

## Summary

`Visualise/Plot_emulator.py` is a post-processing script for plotting emulator outputs and errors.

To run it, you need:
- saved testing outputs (`.pkl` or `.mat`),
- normalization statistics (`normalisation_data.pt`),
- `Nodes.mat` from the MATLAB setup,
- required Python plotting libraries,
- a writable `Figures/` directory.

The provided `output_data/` folder contains results from the best-performing model (40k samples, \(D=400\), 450 epochs) and can be used directly for visualization.