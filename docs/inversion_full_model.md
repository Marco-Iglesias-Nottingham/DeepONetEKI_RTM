# Inversion for Synthetic Experiments Using the MATLAB Resin Infusion Model

`Synthetic.m` generates the synthetic data used in the experiments of Section 3 for both sensor configurations. In this file, the true parameters are specified and the MATLAB resin infusion model is run to produce the true pressure and filling-factor data. Noise is then added to generate the synthetic data files. These files are already provided in the folder `MATLAB_files_for_emulator`.

## `Invert.m`

`Invert.m` is the main routine for the inversion and takes the following inputs:

- `N`: Ensemble size  
- `tag`: Sensor configuration tag  
  - `_100`: case with $M = 100$ sensors  
  - `_for_real`: case with $M = 23$ sensors  

The `_for_real` case is used here for synthetic experiments, but the sensor locations are the same as those used for the real-data experiments.

The largest case used was $N = 5000$, which requires substantial computational resources. In our experiments, we used 90 high-memory cores (see paper) to complete the computation within a couple of hours.

The command

```matlab
Model = set_model_RTM(Dx, Nx, tag);
```

produces the same `Model` settings used in the training-data generation for parameters that are not inferred, such as geometry, resolution, and related fixed settings.

Then,

```matlab
EKI(N, Model, tag, 91882)
```

runs the EKI method described in the paper using the seed adopted in our experiments.

Both prior and posterior parameters are stored in HDF5 files labelled by tag, seed, and ensemble size.

## Stored Variables

The fields are stored in the following variables:

- `Input3`: $L(x)$  
- `Input4`: $\log K_T(x)$  
- `Input5`: $\log K_B(x)$  
- `Input6`: $\xi_T$  
- `Input7`: $\xi_B$  
- `Input9`: $\log K_{\mathrm{def}}(x)$  

The scalar parameters are stored in `Input8`:

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