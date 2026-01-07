# XYModel 项目 README

## 项目简介

`XYModel` 是一个用于模拟 XY 模型的高效计算程序，主要应用于统计物理领域。该项目集成了数据模拟、深度学习模型训练、相变分析和可视化等多种功能，适合研究相变、临界现象和二维系统的磁性行为。

## 项目结构

```
d:/GPU--Version/
├── 数据模拟和生成/          # XY模型数据生成
│   ├── xy_model_simulator_parallel.py    # 并行模拟器
│   ├── run_xy_parallel.py                # 启动脚本
│   └── README_Parallel.md               # 详细文档
├── 模型训练和测试/          # 深度学习模型训练与预测
│   ├── model_clean.py                      # ResNet模型定义
│   ├── model_train_clean.py                # 训练脚本
│   ├── predict_clean.py                    # 预测脚本
│   └── data_analyzer_clean.py              # 数据分析
├── 数据处理和可视化/          # 数据可视化与相变分析
│   ├── xy_model_PhaseData_analysis.py      # 相变数据分析
│   ├── generate_spin_visualizations.py     # 自旋可视化
│   └── classify_png_images.py              # 图片分类
├── 数据归一和极限外推/          # 尺度分析与外推
│   └── 尺度归一和绘图/BKT相变温度外推分析.py
└── 科研绘图/                # 论文级绘图
```

## 功能特性

### 1. 数据模拟
- **并行计算优化**：支持多核并行计算，显著提升模拟效率
- **Swendsen-Wang算法**：高效的蒙特卡洛算法
- **内存优化**：智能内存池管理，减少内存分配开销
- **多种物理量**：计算能量、磁化强度、旋涡数等

### 2. 深度学习模型
- **ResNet架构**：用于相变温度预测
- **温度缩放校准**：优化模型预测概率
- **GPU加速**：支持CUDA训练和推理
- **置信度评估**：提供预测置信度分析

### 3. 数据分析与可视化
- **相变识别**：自动识别有序相和无序相
- **自旋构型可视化**：生成高质量的相图
- **概率曲线分析**：绘制相变概率随温度变化
- **平滑算法**：移动平均、样条插值、LOWESS、Savitzky-Golay

### 4. 尺度分析
- **BKT相变温度外推**：尺度归一分析
- **置信区间估计**：95% CI计算
- **极限外推**：预测无限系统的相变温度

## 安装依赖

```bash
# 基础依赖
pip install numpy matplotlib scipy tqdm

# 深度学习依赖
pip install torch torchvision

# 可选依赖
pip install psutil scikit-learn
```

## 快速开始

### 1. 数据模拟

```bash
cd 数据模拟和生成
python run_xy_parallel.py --lattice-size 32 --temperatures 20 --processes 4
```

### 2. 模型训练

```bash
cd 模型训练和测试
python model_train_clean.py --data-path ../数据处理和可视化/ --epochs 100 --batch-size 64
```

### 3. 预测分析

```bash
cd 模型训练和测试
python predict_clean.py --model best_calibrated_with_temp.pth --input ../数据处理和可视化/相变3/
```

### 4. 数据可视化

```bash
cd 数据处理和可视化
python xy_model_PhaseData_analysis.py
```

## 详细文档

- [并行模拟器文档](数据模拟和生成/README_Parallel.md)
- [技术原理文档](数据模拟和生成/原版/XY模型模拟器技术原理文档.md)
- [操作文档](数据模拟和生成/原版/XY模型模拟器操作文档.md)

## 项目成果

- 成功训练并校准ResNet模型
- 温度缩放参数 T=0.6739
- 50%概率对应的相变温度为 1.092K，误差 0.122K
- 并行计算加速比可达 3.8x（4核）

## 许可证

本项目采用MIT许可证，详见LICENSE文件。