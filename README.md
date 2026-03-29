# DeepONet Emulator for RTM Inversion

This repository contains a DeepONet-based emulator for resin transfer moulding (RTM), together with workflows for:

- data generation (MATLAB full model),
- emulator training,
- emulator testing and uncertainty quantification,
- inversion via Ensemble Kalman Inversion (EKI),
- visualisation of results.

The code supports both **synthetic** and **real-data inversion**, as used in the associated paper.

---

## 🔁 Workflow Overview

The full pipeline is:

1. **Generate training data (MATLAB full model)**
2. **Train DeepONet emulator (`main.py`)**
3. **Test emulator and compute uncertainty (`test_model.py`)**
4. **Run inversion (synthetic or real)**
5. **Visualise outputs**

Each step has detailed documentation in the `docs/` folder.

---

## 📂 Repository Structure

```text
.
├── main.py                     # Training entry point
├── test_model.py               # Emulator testing + UQ
├── EKI_syn.py                  # Synthetic inversion (EKI)
├── EKI_Real.py                 # Real-data inversion (EKI)
├── gather_metrics.py           # Metrics aggregation
├── emulator_tools/             # Model + utilities
│   ├── models.py
│   ├── data_utils.py
│   └── train.py
│
├── MATLAB_code_training_inversion/   # Full RTM simulator + training data generation
├── MATLAB_files_for_emulator/        # Mesh, indices, synthetic data, mass matrix
├── RealData/                         # Real experimental datasets
│
├── Visualise/
│   ├── Plot_Emulator.py              # Plot emulator outputs
│   └── output_data/                  # Example results (best model: 40k, D=400, 450 epochs)
│
├── docs/                             # Detailed documentation
│   ├── emulator_training.md
│   ├── emulator_testing.md
│   ├── InversionDeepONet_synthetic.md
│   ├── InversionDeepOnet_real.md
│   ├── Visualise.md
│   ├── data_generation.md
│   └── inversion_full_model.md
│
└── README.md
```

---

## 🚀 Quick Start

### 1. Train the emulator

```bash
python main.py
```

This creates:

```text
output_<exp_name>/
├── deeponet_epoch_XXX.pt
└── normalisation_data.pt
```

See:  
➡️ `docs/emulator_training.md`

---

### 2. Test the emulator (compute uncertainty)

```bash
python test_model.py \
  --out-name NS40000_D400 \
  --epoch 450 \
  --D 400
```

This produces:

```text
output_<exp_name>/test_outputs_epoch/
└── errors_and_samples_epoch_XXX.pkl
```

These files contain:
- emulator error covariance,
- mean error,
- prediction samples.

See:  
➡️ `docs/emulator_testing.md`

---

### 3. Run inversion

#### Synthetic case

```bash
python EKI_syn.py ...
```

See:  
➡️ `docs/InversionDeepONet_synthetic.md`

---

#### Real-data case

```bash
python EKI_Real.py \
  --exp-name NS40000_D400 \
  --epoch 450 \
  --index-file MATLAB_files_for_emulator/indices_for_real.mat \
  --prior-file Full_EKI_outputs/prior_ensemble_100_NEn5000_seed91882.h5 \
  --file-no 3 \
  --seed 13278 \
  --D 400
```

This produces:
- posterior ensembles,
- predictions,
- summary statistics.

See:  
➡️ `docs/InversionDeepOnet_real.md`

---

### 4. Visualise results

```bash
cd Visualise
python Plot_Emulator.py
```

Uses:
- `.pkl` test outputs,
- normalization stats,
- mesh files (`Nodes.mat`).

See:  
➡️ `docs/Visualise.md`

---

## 📊 Example Outputs

The folder:

```text
Visualise/output_data/
```

contains outputs from the **best-performing model**:

- **40k training samples**
- **latent dimension D = 400**
- **450 epochs**

These include:
- emulator predictions,
- covariance estimates,
- training losses,
- normalization statistics.

---

## 📦 Key Data Dependencies

### Geometry and mesh

```text
MATLAB_files_for_emulator/
├── Nodes.mat
├── MassMatrix.mat
```

- `Nodes.mat` → coordinates (scaled internally to [0,1]²)
- `MassMatrix.mat` → used for physically meaningful error norms

---

### Sensor index sets

```text
indices_for_real.mat
indices_100.mat
```

Used for:
- testing,
- uncertainty quantification,
- inversion.

---

### Synthetic data

```text
syndata_100.mat
syndata_for_real.mat
```

Used in synthetic inversion.

---

### Real data

```text
RealData/
├── Real1.mat ... Real4.mat
├── Inlet1.mat ... Inlet4.mat
```

These define the **four experimental cases** used in the paper.

---

## 🧠 Important Design Notes

### Coordinate scaling

- Spatial coordinates are scaled to `[0,1]²` before entering the network.
- The physical domain is `[0, 0.3]²`.
- Mass-matrix weighting in testing/inversion ensures consistency with the physical domain.

---

### Emulator outputs

The model predicts:
- pressure `p(x,y,t)`
- front `f(x,y,t)`

Pressure is masked by the predicted front during inference.

---

### Uncertainty quantification

- Computed in `test_model.py`
- Stored as covariance + mean error
- Used in inversion to define observation noise

---

## 🔁 Reproducibility

To reproduce results:

You must keep **consistent**:
- training output (`normalisation_data.pt`)
- model checkpoint
- testing `.pkl` file
- node ordering (`Nodes.mat`)
- index file

For the paper results:
- `exp_name = NS40000_D400`
- `epoch = 450`
- `D = 400`
- seed = `13278`
- index set = `indices_for_real.mat`

---

## 📚 Documentation

Detailed guides are available in:

```text
docs/
```

- `emulator_training.md` → training details  
- `emulator_testing.md` → testing + UQ  
- `InversionDeepONet_synthetic.md` → synthetic inversion  
- `InversionDeepOnet_real.md` → real-data inversion  
- `Visualise.md` → plotting and outputs  
- `data_generation.md` → MATLAB pipeline  
- `inversion_full_model.md` → full FEM inversion  


## HPC Environment

All experiments were run using an HPC module-based environment.

The following modules are required:

- GCC/13.2.0
- OpenMPI/4.1.6
- PyTorch 2.4.0 (CUDA 12.4)
- h5py 3.11.0
- scikit-learn 1.4.0

Example setup:

```bash
module purge
module load GCC/13.2.0 OpenMPI/4.1.6
module load PIP-PyTorch/2.4.0-CUDA-12.4.0
module load h5py/3.11.0
module load scikit-learn/1.4.0

---

## 📄 License

See `LICENSE`.

---

## 🧩 Summary

This repository provides a complete pipeline for:

- learning a DeepONet emulator of RTM,
- quantifying emulator uncertainty,
- performing Bayesian-style inversion via EKI,
- applying the method to both synthetic and real experimental data.

For most users, the key scripts are:

- `main.py` → training  
- `test_model.py` → testing + UQ  
- `EKI_syn.py` → synthetic inversion  
- `EKI_Real.py` → real-data inversion  
- `Visualise/Plot_Emulator.py` → plotting  

Start there, and use the `docs/` folder for depth.