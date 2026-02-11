# XYModel 项目 README

## 项目简介

`XYModel` 是一个用于模拟 XY 模型的高效计算程序，主要应用于统计物理领域。该项目集成了数据模拟、深度学习模型训练、相变分析和可视化等多种功能，适合研究相变、临界现象和二维系统的磁性行为。

## 项目结构

```
d:/GPU--Version/
├── 数据模拟和生成/          # XY模型数据生成
│   ├── xy_simulator.py                    # XY模型模拟器核心
│   ├── xy_simulator_core.py               # 模拟器核心类
│   ├── xy_model_simulator_parallel.py     # 并行模拟器
│   ├── xy_parallel.py                     # 并行启动脚本
│   ├── xy_utils.py                        # 工具函数
│   └── test.py                             # 测试脚本
├── 模型训练和测试/          # 深度学习模型训练与预测
│   ├── model.py                            # ResNet模型定义
│   ├── model_train.py                      # 训练脚本
│   ├── predict.py                          # 预测脚本
│   └── plot_probability.py                 # 概率曲线绘制
├── 数据处理和可视化/          # 数据可视化与相变分析
│   ├── xy_model_PhaseData_analysis.py      # 相变数据分析
│   ├── generate_spin_visualizations.py     # 自旋可视化
│   └── classify_png_images.py              # 图片分类
├── 数据归一和极限外推/          # 尺度分析与外推
│   ├── bkt_main.py                         # BKT分析主程序
│   ├── bkt_core.py                         # BKT核心分析
│   ├── bkt_analysis.py                     # BKT分析
│   ├── bkt_plotting.py                     # BKT绘图
│   └── plot_magnetization_comparison.py    # 磁化强度对比
└── 补充/                    # 辅助工具
    ├── auto_train.py                       # 自动训练脚本
    └── analyze_susceptibility.py           # 磁化率分析
```

## 功能特性

### 1. 数据模拟
- **核心模拟器**：基于XY模型的自旋系统模拟
- **并行计算优化**：支持多核并行计算，显著提升模拟效率
- **Swendsen-Wang算法**：高效的蒙特卡洛算法
- **内存优化**：智能内存池管理，减少内存分配开销
- **多种物理量**：计算能量、磁化强度、旋涡数等

### 2. 深度学习模型
- **ResNet架构**：用于相变温度预测
- **GPU加速**：支持CUDA训练和推理

### 3. 数据分析与可视化
- **相变识别**：自动识别有序相和无序相
- **自旋构型可视化**：生成高质量的相图
- **概率曲线分析**：绘制相变概率随温度变化
- **平滑算法**：移动平均、样条插值、LOWESS、Savitzky-Golay

### 4. 尺度分析
- **BKT相变温度外推**：尺度归一分析
- **置信区间估计**：95% CI计算
- **极限外推**：预测无限系统的相变温度
- **磁化强度分析**：不同尺度的磁化行为对比

### 5. 辅助工具
- **自动训练**：简化训练流程的自动化脚本
- **磁化率分析**：精确分析系统磁化率变化

## 安装依赖

```bash
# 基础依赖
pip install numpy matplotlib scipy tqdm

# 深度学习依赖
pip install torch torchvision

# 可选依赖
pip install psutil scikit-learn statsmodels
```

## 快速开始

### 1. 数据模拟

```bash
cd 数据模拟和生成
python xy_simulator.py
```

### 2. 模型训练

```bash
cd 模型训练和测试
python model_train.py
```

### 3. 预测分析

```bash
cd 模型训练和测试
python predict.py --base_path ../数据处理和可视化/test/ --output save_data/
```

### 4. 数据可视化

```bash
cd 数据处理和可视化
python xy_model_PhaseData_analysis.py
```

## 详细文档

项目各模块包含详细注释和使用说明，请参考各目录下的Python脚本文件头部注释。

## 许可证

本项目采用MIT许可证，详见LICENSE文件。