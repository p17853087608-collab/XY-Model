# XYModel 项目 README

## 项目简介

`XYModel` 是一个用于模拟 XY 模型的高效计算程序，主要应用于统计物理领域。该项目通过多种优化策略（如内存池管理、GPU 加速、向量化计算等）显著提高了模拟效率，适合研究相变、临界现象和二维系统的磁性行为。

## 功能特性

- **高效内存管理**：使用 `SmartMemoryPool` 类减少频繁内存分配和释放的开销。
- **三角函数表预计算**：通过 `TrigTable` 类提高三角函数调用效率。
- **并查集优化**：使用 `OptimizedUnionFind` 类加速集群查找。
- **GPU 加速支持**：可选使用 GPU 进行高性能计算。
- **多算法实现**：包含多种蒙特卡洛步进算法，如超优化版本、传统版本和向量化版本。
- **结果可视化**：支持自旋构型的图像保存与可视化。
- **数据保存与加载**：支持模拟结果的持久化存储与读取。

## 安装依赖

在运行项目之前，请确保安装以下依赖库：

- `numpy`
- `cupy`（如需 GPU 支持）
- `scipy`
- `matplotlib`
- `tqdm`

可以通过以下命令安装依赖：

```bash
pip install numpy cupy scipy matplotlib tqdm
```

## 使用方法

1. **导入模块**：

   ```python
   from xy_model_simulator_optimized import XYModelSimulatorOptimized
   ```

2. **初始化模拟器**：

   ```python
   simulator = XYModelSimulatorOptimized(lattice_size=32, equilibrium_steps=500, measurement_steps=5000, use_gpu=True)
   ```

3. **运行模拟**：

   ```python
   results = simulator.run_simulation(temperature_range=(0.1, 2.5), num_temperatures=20)
   ```

4. **可视化结果**：

   ```python
   simulator.plot_results()
   ```

5. **保存与加载数据**：

   ```python
   simulator.save_results("results.txt")
   simulator.save_raw_data("raw_data.npz")
   simulator.load_raw_data("raw_data.npz")
   ```

## 文件说明

- `xy_model_simulator_optimized.py`：核心模拟器实现文件，包含所有优化类和模拟逻辑。
- `test.py`：用于测试模拟器功能的脚本。
- `LICENSE`：项目开源许可证文件。

## 贡献指南

欢迎贡献代码或提出问题。请在 Gitee 上提交 Pull Request 或 Issue。

## 许可证

本项目采用 MIT 许可证。详情请查看 LICENSE 文件。

## 项目主页

本项目托管于 [Gitee](https://gitee.com/Osako2529/XYModel)。欢迎访问项目页面获取最新信息。