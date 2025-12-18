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

## Installation Dependencies

Before running the project, ensure the following dependencies are installed:

- `numpy`
- `cupy` (for GPU support)
- `scipy`
- `matplotlib`
- `tqdm`

Install dependencies using the following command:

```bash
pip install numpy cupy scipy matplotlib tqdm
```

## Usage

1. **Import the module**:

   ```python
   from xy_model_simulator_optimized import XYModelSimulatorOptimized
   ```

2. **Initialize the simulator**:

   ```python
   simulator = XYModelSimulatorOptimized(lattice_size=32, equilibrium_steps=500, measurement_steps=5000, use_gpu=True)
   ```

3. **Run the simulation**:

   ```python
   results = simulator.run_simulation(temperature_range=(0.1, 2.5), num_temperatures=20)
   ```

4. **Visualize results**:

   ```python
   simulator.plot_results()
   ```

5. **Save and load data**:

   ```python
   simulator.save_results("results.txt")
   simulator.save_raw_data("raw_data.npz")
   simulator.load_raw_data("raw_data.npz")
   ```

## File Description

- `xy_model_simulator_optimized.py`: Core simulator implementation containing all optimized classes and simulation logic.
- `test.py`: Script for testing simulator functionality.
- `LICENSE`: Open-source license file for the project.

## Contribution Guidelines

Contributions and issues are welcome. Please submit Pull Requests or Issues on Gitee.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Project Homepage

This project is hosted on [Gitee](https://gitee.com/Osako2529/XYModel). Visit the project page for the latest updates.