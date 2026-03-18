# XYModel Project README

## Project Overview

`XYModel` is an efficient computational program designed to simulate the XY model, primarily applied in the field of statistical physics. This project significantly enhances simulation efficiency through various optimization strategies, such as memory pool management, GPU acceleration, and vectorized computations, making it suitable for studying phase transitions, critical phenomena, and magnetic behavior in two-dimensional systems.

## Features

- **Efficient Memory Management**: Utilizes the `SmartMemoryPool` class to reduce overhead from frequent memory allocation and deallocation.
- **Precomputed Trigonometric Tables**: Improves efficiency of trigonometric function calls via the `TrigTable` class.
- **Optimized Union-Find**: Accelerates cluster detection using the `OptimizedUnionFind` class.
- **GPU Acceleration Support**: Optional GPU-based high-performance computing.
- **Multiple Algorithm Implementations**: Includes various Monte Carlo step algorithms, such as ultra-optimized, traditional, and vectorized versions.
- **Result Visualization**: Supports saving and visualizing spin configurations as images.
- **Data Persistence**: Enables saving and loading of simulation results.

## Project Structure

```
d:/XY Model/
├── 数据模拟和生成/          # XY model data generation
│   ├── xy_simulator.py                    # XY model simulator core
│   ├── xy_simulator_core.py               # Simulator core classes
│   ├── xy_model_simulator_parallel.py     # Parallel simulator
│   ├── xy_parallel.py                     # Parallel launch script
│   ├── xy_utils.py                        # Utility functions
│   └── test.py                             # Test script
├── 模型训练和测试/          # Deep learning model training & prediction
│   ├── model.py                            # ResNet model definition
│   ├── model_train.py                      # Training script
│   ├── predict.py                          # Prediction script
│   └── plot_probability.py                 # Probability curve plotting
├── 数据处理和可视化/          # Data visualization & phase transition analysis
│   ├── xy_model_PhaseData_analysis.py      # Phase transition data analysis
│   ├── generate_spin_visualizations.py     # Spin visualization
│   └── classify_png_images.py              # Image classification
├── 补充/                    # Auxiliary tools
│   ├── auto_train.py                       # Auto training script
│   └── analyze_susceptibility.py           # Susceptibility analysis
├── environment.yml                         # Conda environment configuration
├── requirements.txt                        # Pip dependencies
├── README.md                               # Chinese documentation
├── README.en.md                            # English documentation
└── LICENSE                                 # MIT License
```

## Features

- **Efficient Memory Management**: Utilizes the `SmartMemoryPool` class to reduce overhead from frequent memory allocation and deallocation.
- **Precomputed Trigonometric Tables**: Improves efficiency of trigonometric function calls via the `TrigTable` class.
- **Optimized Union-Find**: Accelerates cluster detection using the `OptimizedUnionFind` class.
- **GPU Acceleration Support**: Optional GPU-based high-performance computing.
- **Multiple Algorithm Implementations**: Includes various Monte Carlo step algorithms, such as ultra-optimized, traditional, and vectorized versions.
- **Result Visualization**: Supports saving and visualizing spin configurations as images.
- **Data Persistence**: Enables saving and loading of simulation results.
- **Deep Learning Integration**: ResNet-based phase transition temperature prediction.
- **Scale Analysis**: BKT phase transition temperature extrapolation with confidence intervals.
- **Auto Training**: Simplified automated training workflow.

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

- Python 3.10.19
- PyTorch 2.9.1 (CUDA 13.0)
- numpy 2.1.2
- matplotlib 3.10.7
- pandas 2.3.3
- scikit-learn 1.7.2
- scipy 1.15.3
- opencv-python 4.12.0.88

## Quick Start

### 1. Data Simulation

```bash
cd 数据模拟和生成
python xy_simulator.py
```

### 2. Model Training

```bash
cd 模型训练和测试
python model_train.py
```

### 3. Prediction Analysis

```bash
cd 模型训练和测试
python predict.py --base_path ../数据处理和可视化/test/ --output save_data/
```

### 4. Data Visualization

```bash
cd 数据处理和可视化
python xy_model_PhaseData_analysis.py
```

### 5. BKT Scale Analysis

```bash
cd 数据归一和极限外推
python bkt_main.py
```

## Project Results

- Successfully trained ResNet model for phase transition temperature prediction
- Parallel computing speedup up to 3.8x (4 cores)

## Contribution Guidelines

Contributions and issues are welcome. Please submit Pull Requests or Issues on Gitee.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Project Homepage

This project is hosted on [Gitee](https://gitee.com/Osako2529/XYModel). Visit the project page for the latest updates.