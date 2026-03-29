# Training data Generation (MATLAB Simulations)

MATLAB code uses libraries from the `Tools` folder.

---

## `training.m`

This is the main script used to generate training data for the DeepONet model by solving the moving boundary problem for different input configurations.

### Main Parameters

- `Dx`: Size of the square computational domain  
- `N_En`: Number of samples to generate  
- `N`: Resolution of the permeability and porosity fields (grid size `N × N`)  

The script includes predefined random seeds used to generate:
- 40,000 samples for training  
- 10,000 samples for testing  

---

### Model Setup

When executed, the script calls:

```matlab
set_model_RTM(Dx, Nx, dataTag)
```

This function has several hardcoded settings, including:
- Output resolution (pressure and front location)  
- Time steps at which outputs are evaluated  

It generates the following files, which must be stored in the root folder `MATLAB_files_for_emulator`:

- `Nodes.mat`: Coordinates of the nodes (used to define the trunk network)  
- `MassMatrix.mat`: Required for computing L2 errors  
- `indices_for_real.mat`, `indices_100.mat`: Used for inversion  

> Note: One of these index files is generated here, but both are recomputed in `Invert.m`. Copies are also provided.

The script ultimately produces a structure: `Model` which contains setting of the forward map for training.

---

### Training Data Generation

The main function for generating training data is:

```matlab
Get_Training_Data(N_En, Model, label, seed)
```

- `label`: Identifier for the dataset (used for reproducibility)  
- `seed`: Random seed controlling sample generation  

---

### Hardcoded Priors

The following parameter ranges are fixed in the code:

```matlab
K_center   = [2.0e-10, 6.5e-10];
K_def      = [0.25e-10, 2.5e-10];
por_center = [0.6, 0.8];
por_def    = [0.55, 0.7];
K_RT       = [20e-10, 50e-10];
por_RT     = [0.9, 0.96];
mu_lim     = [0.085, 0.12];
P_lim      = [92e3, 120e3];
gamma_lim  = [0.6, 1.25];
beta_lim   = [0.2, 0.7];
frac_lim   = [0.35, 0.75];
```

---

### Covariance Definitions

The following covariance matrices define spatial variability:

```matlab
C_LS  = GetCova(Nx, Ny, 0.075, 1,   2);
C_RT  = GetCova(Nx, Ny, 0.1,   0.3, 2);
C_def = GetCova(Nx, Ny, 0.1,   0.3, 2);
```

These correspond to 2D Matérn covariance structures for:
- Level-set function: $L$  
- $\log K_T(\mathbf{x})$  and $\log K_B(\mathbf{x})$  
- $\log K_{\text{def}}(\mathbf{x})$  

Additionally:

```matlab
C_geo = GetCova1D(Nx, 0.15, 0.9*(0.3/20), 1.5);
```

This defines the 1D covariance for race-tracking region widths: $\xi_T$, $\xi_B$.

---
### Parameterisation

```matlab
[perm_for_RTM, poro_for_RTM, ...
    perm_for_deep, poro_for_deep] = get_perm( ...
        perm_C(en), por_C(en), por_def_v(en), ...
        por_RT_top(en), por_RT_bottom(en), ...
        LS_field, Model, ...
        RT_field_top, RT_field_bottom, ...
        RT_top, RT_bottom, ...
        Perm_field_def);
```

This function takes all parameters in the chosen parameterisation and constructs the fields $(\log K(x), \phi(x))$ used as inputs to the resin infusion simulator.

---

### Output Files

The script generates the following files:

- `input_data_<label>.h5`  
  Contains:
  - `Input1`: Samples of $\log K(x)$ defined on a `120 × 120` grid  
  - `Input2`: Samples of $\phi(x)$ defined on a `120 × 120` grid  
  - `Input8`: Samples of the scalar parameters $(\mu, \log P_I, \gamma, \beta, \chi)$  
  *(Note: the numbering is legacy and has been retained for backward compatibility.)*  

- `output_data_<label>.h5`  
  Contains:
  - `Output1`: Pressure, $p(x,t)$  
  - `Output2`: Filling factors, $f(x,t)$  

Outputs are defined on the coordinates provided in `Nodes.mat` and at the time instances hardcoded in `set_model_RTM`.