# XY Model – Monte Carlo + Machine Learning

This repository contains the code and data necessary to reproduce the results of our study on the XY model using Monte Carlo simulations and machine learning.

---

## 📦 Repository Contents

- `example_data/` – Representative example datasets and Monte Carlo simulation scripts
  - `data_generation/` – Monte Carlo simulation scripts
  - `original/` – Original/benchmark implementations
  - `parallel_simulation_results_*/` – Example simulation output data
- `training/` – Deep learning model training scripts
  - `model.py` – ResNet model definition
  - `model_train.py` – Training script
  - `predict.py` – Prediction script
  - `plot_probability.py` – Probability curve plotting
- `analysis/` – Bootstrap analysis, finite-size scaling (FSS), and figure generation
  - `processed_data/` – Processed data used to generate the main figures
    - `fig2/`, `fig3/`, `fig4/` – Data for specific figures
- `data_preprocessing/` – Data preprocessing and spin visualization
  - `generate_spin_visualizations.py` – Spin configuration visualization
  - `classify_png_images.py` – Image classification utilities
  - `xy_model_PhaseData_analysis.py` – Phase transition data analysis
- `model_weights/` – Trained model weights for different lattice sizes
  - `8x8/`, `16x16/`, `32x32/`, `64x64/`, `128x128/`, `256x256/`
- `supplementary/` – Auxiliary tools
  - `auto_train.py` – Automated training script
  - `analyze_susceptibility.py` – Susceptibility analysis

---

## 📊 Data Availability

Due to their large size (tens of GB), the full raw Monte Carlo simulation datasets are **not included** in this repository.

To ensure reproducibility, we provide:
- All scripts required to generate the data (see `example_data/data_generation/`)
- Representative example datasets (see `example_data/` and `analysis/processed_data/`)
- Trained model weights (see `model_weights/`)

The example datasets are intended to illustrate the data format and usage.
The full datasets can be regenerated using the provided scripts.

If needed, the complete raw datasets are available from the authors upon reasonable request.

---

## 🚀 Reproducibility Guide

### 1. Generate Monte Carlo data
```bash
cd example_data/data_generation
python xy_parallel.py
```

### 2. Train the model
```bash
cd training
python model_train.py
```

### 3. Perform bootstrap analysis
```bash
cd analysis
python bkt_analysis_v2.py
```

### 4. Generate figures
```bash
cd data_preprocessing
python generate_spin_visualizations.py
```

---

## 🧪 Quick Start (using example data)

If you do not want to regenerate the full dataset, use the provided example data in `example_data/` and `analysis/processed_data/`.

---

## 🧾 Data Format

The example datasets in `example_data/` demonstrate the structure of the simulation outputs.
Each dataset includes:

* Configuration data (`.npy` spin configuration files)
* Observables
* Labels (if applicable)

Refer to comments in the scripts for detailed format descriptions.

---

## 📌 Notes

* Running full simulations may require significant computational time and storage.
* Results in the paper were obtained using large-scale datasets (tens of GB).

---

## 🔗 Repository

[https://github.com/p17853087608-collab/XY-Model](https://github.com/p17853087608-collab/XY-Model)

---

## Environment Setup

### Option 1: Using Conda (Recommended)

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate XYModel
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv xymodel
source xymodel/bin/activate  # Linux/Mac
xymodel\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- Python 3.10+
- PyTorch 2.x
- numpy
- matplotlib
- pandas
- scikit-learn
- scipy
- opencv-python

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
