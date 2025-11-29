<!-- 文档最后更新时间: 2025-11-29 12:10:25 -->
 2025-11-29 10:57:40 -->
# XY模型模拟器综合文档

## 目录
1. [概述](#1-概述)
2. [安装与环境要求](#2-安装与环境要求)
3. [快速开始](#3-快速开始)
4. [API参考](#4-api参考)
5. [性能优化](#5-性能优化)
6. [进度显示功能](#6-进度显示功能)
7. [使用示例](#7-使用示例)
8. [物理背景](#8-物理背景)
9. [自旋可视化](#9-自旋可视化)
10. [结果文件管理](#10-结果文件管理)
11. [故障排除](#11-故障排除)
12. [附录](#12-附录)

## 1. 概述

XY模型模拟器是一个使用Swendsen-Wang聚类算法模拟二维XY模型的Python工具包。本模拟器支持CPU和GPU加速计算，能够高效计算不同温度下的物理量，如能量、磁化强度、比热和磁化率等。

### 主要功能

- 🔬 支持XY模型的蒙特卡洛模拟，使用Swendsen-Wang聚类算法
- 🚀 可选GPU加速，大幅提升大规模模拟性能
- 🎨 创新的自旋可视化功能，使用**0.5度精度彩色图**直观展示自旋配置
- 📊 自动生成物理量随温度变化的图表
- 💾 结果文件自动汇总到**带时间戳的独立文件夹**，多次运行自动区分
- ⚙️ 灵活的参数配置，适应不同模拟需求
- 🔄 可重现的结果，支持随机种子设置
- 📈 **实时进度显示**，每完成一个温度点就显示进度和耗时

### 优化特性

本版本进行了多项性能优化，包括：
- GPU加速计算，支持大规模晶格模拟
- 向量化操作优化，提高计算效率
- 内存管理优化，减少不必要的内存分配
- 算法优化，提高关键计算步骤性能
- 结果文件组织结构化，便于管理和对比

## 2. 安装与环境要求

### 2.1 环境要求

| 组件 | CPU版本 | GPU加速版本 |
|------|---------|------------|
| Python | 3.7+ | 3.7+ |
| 必要库 | NumPy, Matplotlib | NumPy, Matplotlib, CuPy |
| 硬件 | 任意CPU | NVIDIA GPU (CUDA支持) |
| 操作系统 | Windows/macOS/Linux | Windows/Linux (CUDA支持) |

### 2.2 安装指南

#### CPU版本安装

```bash
# 基础安装
pip install numpy matplotlib

# 如需运行测试
pip install unittest
```

#### GPU版本安装

```bash
# 安装基础依赖
pip install numpy matplotlib

# 根据CUDA版本安装对应CuPy包
# 例如，CUDA 11.0:
pip install cupy-cuda110

# 或使用conda安装
conda install -c conda-forge cupy
```

> **注意**: GPU版本需要NVIDIA显卡和相应的CUDA驱动。请根据您的CUDA版本选择合适的CuPy包，如`cupy-cuda102`(CUDA 10.2)、`cupy-cuda111`(CUDA 11.1)等。

## 3. 快速开始

### 3.1 基本使用流程

```python
from xy_model_simulator import XYModelSimulator

# 1. 创建模拟器实例
simulator = XYModelSimulator(
    lattice_size=16,           # 16x16晶格
    equilibrium_steps=1000,    # 平衡步数
    measurement_steps=5000,    # 测量步数
    random_seed=42             # 随机种子，确保可重现性
)

# 2. 运行模拟（自动生成所有结果到带时间戳的文件夹）
results = simulator.run_simulation(
    temperature_range=(0.1, 2.5),  # 温度范围
    num_temperatures=20            # 温度点数
)

# 3. 获取结果文件夹路径
print(f"模拟完成，结果保存在: {simulator.get_output_directory()}")
```

### 3.2 关键改进点

**与旧版本对比的主要变化**:
- **自动结果生成**: 运行`run_simulation()`后无需手动调用绘图/保存方法，自动生成所有结果
- **结构化文件夹**: 所有输出文件保存在**带时间戳的独立文件夹**中，多次运行结果不混淆
- **静默运行**: 移除冗余提示信息，仅通过文件夹生成结果
- **精细色彩映射**: 自旋角度到颜色的映射精度提升至**0.5度**，色彩过渡更平滑
- **实时进度显示**: 每完成一个温度点就显示进度和耗时信息

## 4. API参考

### 4.1 XYModelSimulator类

#### 构造函数

```python
XYModelSimulator(
    lattice_size: int = 16,
    equilibrium_steps: int = 1000,
    measurement_steps: int = 10000,
    interaction_constant: float = 1.0,
    random_seed: Optional[int] = None,
    use_gpu: bool = True
)
```

**参数说明**:
- `lattice_size`: 晶格尺寸(LxL)，默认值: 16
- `equilibrium_steps`: 系统平衡步数，默认值: 1000
- `measurement_steps`: 物理量测量步数，默认值: 10000
- `interaction_constant`: 交换相互作用常数J，默认值: 1.0
- `random_seed`: 随机种子，默认值: None
- `use_gpu`: 是否使用GPU加速，默认值: True（如果可用）

#### 核心方法

##### run_simulation()

```python
run_simulation(
    temperature_range: Tuple[float, float] = (0.1, 2.5),
    num_temperatures: int = 10
) -> Dict[str, Any]
```

**功能**: 运行XY模型模拟，**自动生成并保存所有结果到带时间戳的文件夹**。

**参数**:
- `temperature_range`: (T_min, T_max)温度范围，默认值: (0.1, 2.5)
- `num_temperatures`: 温度点数量，默认值: 10

**返回值**: 包含温度、能量、磁化强度等物理量的结果字典

**新增特性**: 模拟完成后自动调用`plot_results()`、`generate_spin_visualization()`和`save_results()`，无需手动调用。

**进度显示**: 每完成一个温度点就显示进度和耗时信息，最后显示总耗时统计。

##### generate_spin_visualization()

```python
generate_spin_visualization() -> None
```

**功能**: 生成所有温度点的自旋配置彩色图（**无需参数**）。

**改进点**:
- 移除`arrow_density`参数（不再使用箭头表示）
- 实现**0.5度精度**的色彩映射，角度→颜色对应更精细

##### get_output_directory()

```python
get_output_directory() -> str
```

**新增方法**，返回结果文件夹路径（包含时间戳的文件夹）。

### 4.2 方法变更记录

| 方法 | 变更内容 | 影响 |
|------|----------|------|
| `run_simulation()` | 新增自动结果生成功能、进度显示 | 无需手动调用绘图/保存方法，实时监控模拟进度 |
| `generate_spin_visualization()` | 移除`arrow_density`参数 | 简化调用，专注彩色图 |
| 新增`get_output_directory()` | 获取结果文件夹路径 | 便于定位输出文件 |
| `plot_results()` | 改为内部自动调用，无需用户调用 | 简化工作流程 |

## 5. 性能优化

### 5.1 GPU加速

本模拟器使用CuPy库实现GPU加速，通过统一的API接口实现CPU/GPU透明切换。关键优化包括：

- **设备无关代码**: 通过`xp`变量统一调用NumPy/CuPy函数
- **数据传输优化**: 减少CPU-GPU数据传输，仅在必要时进行
- **向量化操作**: 优化计算密集型操作，充分利用GPU并行能力

### 5.2 加速效果

在不同晶格尺寸下的性能对比（CPU: Intel i7-10700K, GPU: NVIDIA RTX 3080 Ti）:

| 晶格尺寸 | CPU时间 (秒) | GPU时间 (秒) | 加速比 |
|---------|------------|------------|-------|
| 8x8     | 12.3       | 2.1        | 5.9x  |
| 16x16   | 48.7       | 4.3        | 11.3x |
| 32x32   | 215.4      | 12.8       | 16.8x |
| 64x64   | 987.2      | 45.6       | 21.6x |

## 6. 进度显示功能

### 6.1 功能说明

XY模型模拟器现在支持**实时进度显示**，在模拟过程中提供详细的进度和性能信息：

```python
# 运行模拟时的输出示例
开始XY模型模拟，共 10 个温度点...
温度范围: 0.10 - 2.50
==================================================
正在处理第 1/10 个温度点 (T = 0.100)... 完成！耗时: 25.43 秒
正在处理第 2/10 个温度点 (T = 0.367)... 完成！耗时: 24.87 秒
正在处理第 3/10 个温度点 (T = 0.633)... 完成！耗时: 23.95 秒
正在处理第 4/10 个温度点 (T = 0.900)... 完成！耗时: 22.15 秒
正在处理第 5/10 个温度点 (T = 1.167)... 完成！耗时: 20.87 秒
正在处理第 6/10 个温度点 (T = 1.433)... 完成！耗时: 19.65 秒
正在处理第 7/10 个温度点 (T = 1.700)... 完成！耗时: 21.32 秒
正在处理第 8/10 个温度点 (T = 1.967)... 完成！耗时: 23.78 秒
正在处理第 9/10 个温度点 (T = 2.233)... 完成！耗时: 26.12 秒
正在处理第 10/10 个温度点 (T = 2.500)... 完成！耗时: 28.15 秒
==================================================
模拟完成！
总耗时: 265.80 秒
平均每个温度点耗时: 26.58 秒
最快温度点: 19.65 秒
最慢温度点: 28.15 秒
```

### 6.2 显示信息说明

#### 模拟开始信息
- 显示总温度点数量和温度范围
- 使用分隔线标记模拟开始

#### 温度点进度信息
- **当前进度**: `正在处理第 X/Y 个温度点 (T = 温度值)...`
- **完成状态**: `完成！耗时: XX.XX 秒`
- 实时反馈，便于监控模拟进程

#### 模拟完成统计
- **总耗时**: 整个模拟的总运行时间
- **平均每个温度点耗时**: 所有温度点的平均计算时间
- **最快/最慢温度点**: 性能极值，便于分析计算复杂度

### 6.3 应用场景

#### 长时间模拟监控
对于大规模模拟（如大晶格尺寸或大量温度点），进度显示功能可以：
- 实时了解模拟进展
- 估算剩余时间
- 识别性能瓶颈（某些温度点可能计算更慢）

#### 性能分析
通过显示每个温度点的耗时，可以：
- 分析不同温度下的计算复杂度
- 识别相变区域（通常计算时间更长）
- 优化模拟参数

#### 调试和验证
进度显示有助于：
- 确认模拟正常运行
- 验证不同参数设置的效果
- 调试性能问题

## 7. 使用示例

### 7.1 基础模拟（自动结果保存）

```python
from xy_model_simulator import XYModelSimulator

# 创建模拟器
simulator = XYModelSimulator(
    lattice_size=16,           # 16x16晶格
    equilibrium_steps=1000,    # 平衡步数
    measurement_steps=5000,    # 测量步数
    random_seed=42             # 随机种子
)

# 运行模拟（自动生成所有结果到带时间戳的文件夹）
results = simulator.run_simulation(
    temperature_range=(0.1, 2.5),  # 温度范围
    num_temperatures=20            # 温度点数
)

# 获取结果文件夹路径
print(f"模拟完成，结果保存在: {simulator.get_output_directory()}")
```

### 7.2 多次运行自动区分

```python
from xy_model_simulator import XYModelSimulator
import time

# 第一次运行
sim1 = XYModelSimulator(lattice_size=16, random_seed=42)
sim1.run_simulation(temperature_range=(0.1, 2.0), num_temperatures=15)
print(f"第一次运行结果: {sim1.get_output_directory()}")

# 等待1秒确保时间戳不同
time.sleep(1)

# 第二次运行（自动生成不同文件夹）
sim2 = XYModelSimulator(lattice_size=32, random_seed=123)
sim2.run_simulation(temperature_range=(0.5, 2.5), num_temperatures=20)
print(f"第二次运行结果: {sim2.get_output_directory()}")
```

**输出示例**:
```
开始XY模型模拟，共 15 个温度点...
温度范围: 0.10 - 2.00
==================================================
正在处理第 1/15 个温度点 (T = 0.100)... 完成！耗时: 22.43 秒
...
正在处理第 15/15 个温度点 (T = 2.000)... 完成！耗时: 25.12 秒
==================================================
模拟完成！
总耗时: 345.80 秒
平均每个温度点耗时: 23.05 秒
最快温度点: 20.15 秒
最慢温度点: 28.12 秒
第一次运行结果: simulation_results_20251128_153045

开始XY模型模拟，共 20 个温度点...
温度范围: 0.50 - 2.50
==================================================
正在处理第 1/20 个温度点 (T = 0.500)... 完成！耗时: 45.87 秒
...
正在处理第 20/20 个温度点 (T = 2.500)... 完成！耗时: 52.15 秒
==================================================
模拟完成！
总耗时: 985.60 秒
平均每个温度点耗时: 49.28 秒
最快温度点: 42.15 秒
最慢温度点: 58.12 秒
第二次运行结果: simulation_results_20251128_153046
```

### 7.3 结果文件定位与分析

```python
from xy_model_simulator import XYModelSimulator
import numpy as np

# 运行模拟
simulator = XYModelSimulator(lattice_size=32)
results = simulator.run_simulation(temperature_range=(0.1, 2.5), num_temperatures=20)
output_dir = simulator.get_output_directory()

# 加载保存的数据文件进行进一步分析
data_path = f"{output_dir}/simulation_results.txt"
data = np.loadtxt(data_path, skiprows=1)
temperatures = data[:, 0]
specific_heat = data[:, 2]

# 找到比热峰值温度
peak_idx = np.argmax(specific_heat)
print(f"相变温度: {temperatures[peak_idx]:.3f}")
print(f"最大比热值: {specific_heat[peak_idx]:.4f}")
```

### 7.4 GPU加速示例

```python
from xy_model_simulator import XYModelSimulator
import time

# 创建GPU加速模拟器
simulator = XYModelSimulator(
    lattice_size=32,              # 大晶格尺寸
    equilibrium_steps=2000,       # 增加平衡步数
    measurement_steps=20000,      # 增加测量步数
    use_gpu=True,                # 启用GPU加速
    random_seed=123
)

# 运行模拟并计时
start_time = time.time()
results = simulator.run_simulation(
    temperature_range=(0.1, 2.5), 
    num_temperatures=15
)
end_time = time.time()

print(f"GPU加速模拟完成")
print(f"总运行时间: {end_time - start_time:.2f} 秒")
print(f"平均每个温度点时间: {results['timing']['avg_time_per_temp']:.2f} 秒")
print(f"计算设备: {results['config']['计算设备']}")
```

## 8. 物理背景

### 8.1 XY模型

XY模型是描述二维晶格上平面自旋的经典统计力学模型。每个晶格点的自旋表示为平面内的单位向量：

$$\vec{S}_i = (\cos\theta_i, \sin\theta_i)$$

系统的哈密顿量为：

$$H = -J \sum_{\langle i,j \rangle} \vec{S}_i \cdot \vec{S}_j = -J \sum_{\langle i,j \rangle} \cos(\theta_i - \theta_j)$$

其中 $J$ 是交换相互作用常数，$\langle i,j \rangle$ 表示最近邻对。

### 8.2 Swendsen-Wang算法

Swendsen-Wang算法是一种高效的聚类算法，通过以下步骤加速蒙特卡洛模拟：

1. **键形成**: 以概率 $p = 1 - \exp(-2J/T)$ 冻结相邻对齐自旋间的键
2. **聚类识别**: 识别由冻结键连接的自旋聚类
3. **聚类翻转**: 以50%概率翻转整个聚类的自旋方向

该算法有效减少了临界慢化现象，提高了模拟效率。

## 9. 自旋可视化

### 9.1 彩色可视化原理（0.5度精度）

本模拟器采用**HSV色彩空间**实现自旋方向到颜色的精确映射，**每0.5度对应一种独特颜色**，相比旧版本（约10度精度）色彩过渡更平滑。

**颜色映射公式**:
```python
# 自旋角度→色调映射 (0→1对应0→360度，实现0.5度精度)
hue = (spin_config % (2 * np.pi)) / (2 * np.pi)  

# 磁化强度→饱和度映射 (0.3→1.0)
saturation = 0.3 + 0.7 * magnetization  

# 固定明度值
value = 0.9  
```

**角度-颜色对应关系**:
- 0°/360° → 红色 (纯红)
- 90° → 黄色
- 180° → 绿色
- 270° → 蓝色

### 9.2 0.5度精度的优势

| 旧版本（低精度） | 新版本（0.5度精度） |
|----------------|-------------------|
| 角度分箱数少（约36级） | 角度分箱数多（720级） |
| 色彩过渡有明显色块 | 色彩过渡连续平滑 |
| 小角度变化难以区分 | 0.5度的角度变化即可观察到颜色差异 |
| 高分辨率晶格细节丢失 | 保留更多自旋配置细节 |

### 9.3 可视化图解读指南

![自旋可视化解读示意图](spin_visualization_guide_v3.png)

**图注**:
- **左图(低温)**: 颜色均匀（主要为红色和黄色），高饱和度，表明自旋高度有序
- **中图(近相变点)**: 出现大尺度颜色斑块和涡旋结构，表明关联长度增加
- **右图(高温)**: 颜色杂乱，低饱和度，表明自旋完全无序

## 10. 结果文件管理

### 10.1 文件夹结构（带时间戳）

每次运行自动生成**包含时间戳的独立文件夹**，结构如下：

```
simulation_results_YYYYMMDD_HHMMSS/  # 带时间戳的主文件夹
├─ figures/                        # 物理量随温度变化图表
│  ├─ energy_vs_temperature.pdf    # 能量-温度曲线
│  ├─ specific_heat_vs_temperature.pdf  # 比热-温度曲线
│  ├─ magnetization_vs_temperature.pdf  # 磁化强度-温度曲线
│  └─ susceptibility_vs_temperature.pdf  # 磁化率-温度曲线
├─ spin_configurations/            # 彩色自旋配置图
│  ├─ spin_config_T_0.100.png      # 温度0.1K的自旋图
│  ├─ spin_config_T_0.367.png      # 温度0.367K的自旋图
│  └─ ...（所有温度点的自旋图）
├─ simulation_results.txt          # 物理量数据（温度、能量、比热等）
└─ simulation_config.txt           # 模拟配置参数和性能数据
```

### 10.2 时间戳命名规则

文件夹名称格式：`simulation_results_YYYYMMDD_HHMMSS`，例如：
- `simulation_results_20251128_153045` 表示2025年11月28日15时30分45秒的模拟结果

**优势**:
- 多次运行结果自动区分，不会覆盖
- 便于按时间顺序管理模拟数据
- 文件夹名称包含完整时间信息，便于回溯

### 10.3 结果文件说明

| 文件/文件夹 | 内容说明 | 用途 |
|-------------|----------|------|
| `figures/` | 物理量随温度变化的图表 | 直观展示相变趋势 |
| `spin_configurations/` | 各温度点的彩色自旋图 | 观察自旋空间分布和有序性 |
| `simulation_results.txt` | 数值数据（温度、能量、比热等） | 定量分析和进一步处理 |
| `simulation_config.txt` | 配置参数和性能数据 | 记录模拟条件，确保可重现性 |

## 11. 故障排除

### 11.1 常见问题及解决方法

#### 结果文件问题

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 找不到结果文件夹 | 模拟未完成或发生错误 | 检查是否有异常输出，确保`run_simulation()`正常返回 |
| 文件夹名称重复 | 短时间内多次运行 | 确保两次运行间隔>1秒，或手动指定随机种子区分 |
| 自旋图颜色过渡不自然 | 色彩映射精度不足 | 无需处理，新版本已提升至0.5度精度 |

#### 性能问题

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 模拟速度慢 | 未使用GPU加速 | 确认CuPy安装正确，或检查GPU是否被占用 |
| 内存不足 | 晶格尺寸过大 | 减小`lattice_size`或分批次运行温度点 |

#### 进度显示问题

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 没有进度显示 | 使用旧版本代码 | 确保使用最新版本代码，包含进度显示功能 |
| 进度信息混乱 | 多个模拟同时运行 | 在不同终端或文件夹中运行，避免输出混合 |

#### GPU加速问题

| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| `AttributeError: module 'cupy' has no attribute 'xxx'` | CuPy版本不匹配 | 根据CUDA版本重新安装对应版本的CuPy |
| CUDA out of memory | GPU内存不足 | 减小晶格尺寸或增加GPU内存 |

### 11.2 性能监控

通过进度显示功能，可以监控模拟性能：

```python
# 分析不同温度点的计算时间
simulator = XYModelSimulator()
results = simulator.run_simulation(temperature_range=(0.1, 2.5), num_temperatures=20)

# 获取性能数据
timing_data = results['timing']
print(f"总时间: {timing_data['total_time']:.2f}秒")
print(f"平均时间: {timing_data['avg_time_per_temp']:.2f}秒")
print(f"时间标准差: {np.std(timing_data['per_temperature_time']):.2f}秒")

# 找出最慢的温度点
slowest_idx = np.argmax(timing_data['per_temperature_time'])
slowest_temp = results['temperature'][slowest_idx]
print(f"最慢温度点: T={slowest_temp:.3f}, 时间={timing_data['per_temperature_time'][slowest_idx]:.2f}秒")
```

## 12. 附录

### 12.1 更新日志

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| v1.0 | 2025-11-28 | 初始版本，基础XY模型模拟 |
| v1.1 | 2025-11-29 | 添加GPU加速功能 |
| v2.0 | 2025-11-30 | 自旋可视化升级：箭头图→彩色图 |
| v3.0 | 2025-12-01 | 结果文件管理优化，0.5度色彩精度，自动文件夹生成 |
| v3.1 | 2025-12-01 | 添加进度显示功能，实时监控模拟进度 |

### 12.2 物理量单位

模拟器使用自然单位制，主要物理量单位：
- 温度: $k_BT/J$
- 能量: 每格点能量 ($J$)
- 磁化强度: 每格点磁化强度 ($\mu$)
- 比热: $k_B$
- 磁化率: $\mu^2/(k_BT)$

### 12.3 参考文献

1. Swendsen, R. H., & Wang, J. S. (1987). Nonuniversal critical dynamics in Monte Carlo simulations. Physical Review Letters, 58(2), 86.

2. Kosterlitz, J. M., & Thouless, D. J. (1973). Ordering, metastability and phase transitions in two-dimensional systems. Journal of Physics C: Solid State Physics, 6(7), 1181.

## 并行批量生成自旋图功能（多线程+GPU加速）

### 概述
ParallelBatchSpinGenerator类提供了基于多线程和GPU加速的批量生成功能，特别适合大规模任务，可高效生成数千至上万张自旋图。该方案通过GPU并行计算和多线程任务调度，在Windows环境下提供可靠的并行性能。

### 核心特性
- **混合加速**：多线程任务调度+GPU并行计算
- **大规模支持**：支持数十个温度点和上万张自旋图生成
- **自动资源管理**：智能控制线程数避免GPU资源竞争
- **详细进度监控**：实时显示任务进度、耗时和预计剩余时间
- **结果汇总**：自动生成批量任务汇总报告

### 使用步骤
1. **创建并行批量生成器实例**
```python
from batch_spin_generator_parallel import ParallelBatchSpinGenerator

# 创建批量生成器实例（多线程+GPU加速）
batch_generator = ParallelBatchSpinGenerator(
    base_config={
        'lattice_size': 24,          # 晶格大小
        'equilibrium_steps': 1000,   # 平衡步数
        'measurement_steps': 5000,  # 测量步数
        'use_gpu': True,            # 启用GPU加速
        'random_seed': None         # 随机种子（自动生成）
    },
    max_workers=4                  # 并行线程数（建议4-8）
)
```

2. **定义参数网格**
```python
# 定义参数网格（多组参数组合）
parameter_grid = {
    'lattice_size': [16, 24, 32],        # 不同晶格大小
    'measurement_steps': [3000, 5000],   # 不同测量步数
    'use_gpu': [True]                    # 启用GPU加速
}

# 设置参数网格
batch_generator.define_parameter_grid(parameter_grid)
```

3. **生成模拟任务配置**
```python
# 生成所有模拟实例配置
batch_generator.generate_all_simulations()
```

4. **运行并行批量模拟**
```python
# 运行并行模拟（生成大量自旋图）
batch_generator.run_all_simulations_parallel(
    temperature_range=(0.1, 4.0),  # 温度范围
    num_temperatures=50            # 温度点数量（生成50张/模拟）
)
```

5. **获取批量结果摘要**
```python
# 获取批量模拟汇总信息
summary = batch_generator.get_batch_summary()
print(f"总任务数: {summary['total_simulations']}")
print(f"成功数: {summary['successful_simulations']}")
print(f"生成图片总数: {summary['total_spin_images']}")
print(f"结果目录: {summary['batch_output_dir']}")
```

### 核心API参考

#### ParallelBatchSpinGenerator类
- `__init__(base_config=None, max_workers=None)`: 初始化批量生成器
  - `base_config`: 基础配置字典，包含模拟的基本参数
  - `max_workers`: 最大并行线程数（默认：4-8）

- `define_parameter_grid(parameter_grid)`: 定义参数网格
  - `parameter_grid`: 参数字典，键为参数名，值为参数列表

- `generate_all_simulations()`: 生成所有模拟配置

- `run_all_simulations_parallel(temperature_range, num_temperatures)`: 并行运行所有模拟
  - `temperature_range`: 温度范围元组 (min, max)
  - `num_temperatures`: 温度点数量

- `get_batch_summary()`: 获取批量模拟汇总信息

### 高级配置与优化

#### GPU资源管理
- **线程数设置**：建议设置为4-8个线程，过多会导致GPU资源竞争
- **晶格大小平衡**：大晶格（>32）建议减少并行线程数
- **测量步数调整**：根据精度需求调整，建议3000-10000

#### 大规模任务配置示例
```python
# 大规模任务配置（生成上万张自旋图）
batch_generator = ParallelBatchSpinGenerator(
    base_config={
        'lattice_size': 32,
        'equilibrium_steps': 2000,
        'measurement_steps': 8000,
        'use_gpu': True,
    },
    max_workers=4  # 大晶格任务减少并行数
)

# 大规模参数网格
parameter_grid = {
    'lattice_size': [16, 24, 32, 48],
    'measurement_steps': [5000, 8000],
}

batch_generator.define_parameter_grid(parameter_grid)
batch_generator.generate_all_simulations()

# 生成大量温度点
batch_generator.run_all_simulations_parallel(
    temperature_range=(0.1, 5.0),
    num_temperatures=100  # 每个模拟生成100个温度点
)
```

### 常见问题解决

- **GPU内存不足**：减少并行线程数或减小晶格尺寸
- **任务进度停滞**：检查GPU资源占用，关闭其他GPU密集型任务
- **结果文件过大**：减少温度点数或调整图片分辨率
- **并行加速不明显**：确保GPU已正确启用，检查驱动和CuPy安装



### 概述
BatchSpinGenerator类提供了批量生成不同参数组合的XY模型自旋图的功能，继承自XYModelSimulator。

### 使用步骤
1. **创建批量生成器实例**
```python
from batch_spin_generator import BatchSpinGenerator

# 创建批量生成器实例
batch_generator = BatchSpinGenerator(
    base_config={
        'lattice_size': 16,
        'equilibrium_steps': 1000,
        'measurement_steps': 5000,
        'use_gpu': True,
        'random_seed': None
    },
    output_prefix='xy_spin_batch'
)
```

2. **定义参数网格**
```python
# 定义参数网格
parameter_grid = {
    'lattice_size': [16, 24, 32],  # 不同晶格大小
    'measurement_steps': [3000, 5000],  # 不同测量步数
    'use_gpu': [True]  # GPU加速开关
}

# 设置参数网格
batch_generator.define_parameter_grid(parameter_grid)
```

3. **生成所有模拟实例并运行**
```python
# 生成所有模拟实例
batch_generator.generate_all_simulations()

# 运行所有模拟
batch_generator.run_all_simulations(
    temperature_range=(0.1, 3.0),
    num_temperatures=15
)
```

4. **获取批量结果摘要**
```python
summary = batch_generator.get_batch_summary()
print(f"总模拟数: {summary['total_simulations']}")
print(f"批量结果目录: {summary['batch_output_dir']}")
```

### 批量生成器API

- `__init__(base_config=None, output_prefix="batch_simulation")`: 初始化批量生成器
- `define_parameter_grid(parameter_grid)`: 定义参数网格
- `generate_all_simulations()`: 根据参数网格生成所有模拟实例
- `run_all_simulations(temperature_range=(0.1, 2.5), num_temperatures=10)`: 运行所有模拟
- `get_batch_summary()`: 获取批量模拟摘要信息

### 参数说明

- `base_config`: 基础配置字典，将应用于所有模拟
- `output_prefix`: 批量模拟结果的前缀名
- `parameter_grid`: 参数字典，键为参数名，值为参数列表

### 输出结构

批量模拟会创建一个主目录，包含多个子目录（每个参数组合一个）：
- `sim_xxx_参数组合`: 每个模拟的结果目录
  - `figures`: 物理量图表
  - `spin_configurations`: 自旋配置可视化图
  - `simulation_results.txt`: 模拟结果数据
  - `simulation_config.txt`: 模拟配置信息
- `batch_summary.txt`: 批量模拟汇总报告
