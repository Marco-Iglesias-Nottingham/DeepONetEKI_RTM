# DeepONet Emulator for RTM Inversion

This repository contains a DeepONet-based emulator for resin transfer moulding (RTM), together with workflows for:

- data generation (MATLAB full model),
- emulator training,
- emulator testing and uncertainty quantification,
- inversion via Ensemble Kalman Inversion (EKI),
- visualisation of results.

The code supports both **synthetic** and **real-data inversion**, as used in the associated paper.

---

## 📄 Associated paper

This repository accompanies the following manuscript:

**DeepONet-Accelerated Bayesian Inversion for Moving Boundary Problems**  
[arXiv:2512.20268](https://arxiv.org/abs/2512.20268)

*(Under review at Computer Methods in Applied Mechanics and Engineering)*

---

## 🔁 Workflow Overview

The full pipeline is:

1. **Generate training data (MATLAB full model)**
2. **Train DeepONet emulator (`main.py`)**
3. **Test emulator and compute emulator error mean and covariance (`test_model.py`)**
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

See:  [docs/emulator_training.md](docs/emulator_training.md)

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
➡️ [docs/emulator_testing.md](docs/emulator_testing.md)

---

### 3. Run inversion

#### Synthetic case

```bash
python EKI_syn.py ...
```

See:  
➡️ [docs/InversionDeepONet_synthetic.md](docs/InversionDeepONet_synthetic.md)

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
➡️ [docs/InversionDeepOnet_real.md](docs/InversionDeepOnet_real.md)

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
➡️ [docs/Visualise.md](docs/Visualise.md)

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

---

## 📦 Data Availability

The datasets, trained models, and inversion results used in this work are available on Zenodo:

👉 https://doi.org/10.5281/zenodo.19318311  

**Concept DOI (all versions):** https://doi.org/10.5281/zenodo.19318309  

If you use this data, please cite the Zenodo record.

---

### Dataset contents and notes

**1. Training data (not included)**  
The full training datasets:
- `input_data_batch_seed_1.h5`  
- `output_data_batch_seed_1.h5`  

are **not provided** due to their large size (~40k samples).  

However, full reproducibility is ensured:
- MATLAB code for data generation is included in this repository  
- Random seeds are provided  
- See: [docs/data_generation.md](docs/data_generation.md)  

---

**2. Provided input–output dataset (10k samples)**  
- `input_data_batch_seed_2.h5`  
- `output_data_batch_seed_2.h5`  

These contain **10,000 input–output pairs**, generated independently from the training data.

They follow the same format as the training dataset and can be:
- used for testing (as in this work), or  
- repurposed for training by relabelling if desired  

---

**3. Prior and posterior ensembles (full model)**  
- Prior and posterior ensembles obtained using the full-order model are included  

These are provided to:
- enable direct comparison with surrogate-based inversion  
- avoid the need to run computationally expensive full-model inversion  

The full-model inversion code (MATLAB) is also provided in this repository.  
See: [Full-model inversion documentation](docs/inversion_full_model.md)

---

**4. Trained DeepONet models and results**  

Zenodo includes trained DeepONet models (450 epochs) for:

- 10k training samples  
- 20k training samples  
- 40k training samples  

These correspond to the experiments reported in the paper.

Also included:
- emulator test metrics  
- uncertainty quantification outputs (mean and covariance)  
- posterior ensembles from DeepONet-EKI (synthetic and real data)  

---

### Reproducibility

All results in the paper can be reproduced using:
- this repository (code, licensed under the MIT License),
- the Zenodo dataset (data and trained models),
- and the provided documentation in `docs/`.

## 🔁 Reproducing the Published Results

To reproduce the results reported in the paper, the following must be kept **consistent**:

- training normalization (`normalisation_data.pt`)
- model checkpoint (`deeponet_epoch_XXX.pt`)
- testing outputs (`.pkl` files)
- node ordering (`Nodes.mat`)
- sensor index file

For the main results:

- `exp_name = NS40000_D400`  
- `epoch = 450`  
- `D = 400`  
- seed = `13278`  
- index set = `indices_for_real.mat`  

The corresponding trained models, test outputs, and inversion results are available on Zenodo (see Data Availability section).

## 🧠 Important Design Notes

### Emulator outputs

The model predicts:
- pressure `p(x,y,t)`
- front `f(x,y,t)`

Pressure is masked by the predicted front during inference.

---

### Emulator uncertainty

- Computed in `test_model.py`
- Stored as covariance + mean error
- Used in inversion (enhanced model error)

---



## 📚 Documentation

Detailed guides are available in:

```text
docs/
```

- `data_generation.md` → MATLAB pipeline  
- `inversion_full_model.md` → full FEM inversion
- `emulator_training.md` → training details  
- `emulator_testing.md` → testing + enhanced model error stats 
- `InversionDeepONet_synthetic.md` → synthetic inversion  
- `InversionDeepOnet_real.md` → real-data inversion  
- `Visualise.md` → plotting and outputs  
  


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