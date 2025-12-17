# XY模型模拟器操作文档

## 📋 概述

`xy_model_simulator_optimized.py` 是一个高性能的2D XY模型蒙特卡洛模拟器，使用Swendsen-Wang聚类算法进行优化。该模拟器支持多种优化特性，包括智能内存管理、向量化计算和自适应算法选择。

---

## 🚀 快速开始

### 基本使用

```python
import xy_model_simulator_optimized as xy_sim

# 创建模拟器实例
simulator = xy_sim.XYModelSimulator(
    lattice_size=16,           # 晶格尺寸 16x16
    equilibrium_steps=1000,     # 平衡步数
    measurement_steps=10000,    # 测量步数
    random_seed=42              # 随机种子
)

# 运行模拟
results = simulator.run_simulation(
    temperature_range=(0.5, 2.5),  # 温度范围
    num_temperatures=10              # 温度点数量
)

# 获取结果
print(f"能量范围: [{results['energy'].min():.3f}, {results['energy'].max():.3f}]")
print(f"磁化强度范围: [{results['magnetization'].min():.3f}, {results['magnetization'].max():.3f}]")
```

---

## ⚙️ 核心参数配置

### 构造函数参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `lattice_size` | int | 16 | 晶格尺寸 L×L |
| `equilibrium_steps` | int | 1000 | 系统平衡蒙特卡洛步数 |
| `measurement_steps` | int | 10000 | 物理量测量步数 |
| `interaction_constant` | float | 1.0 | 相互作用常数J |
| `random_seed` | int \| None | None | 随机种子 |
| `use_gpu` | bool | True | 是否使用GPU加速（当前版本仅CPU） |

### 模拟运行参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `temperature_range` | tuple[float, float] | (0.1, 2.5) | 温度范围 (T_min, T_max) |
| `num_temperatures` | int | 10 | 温度点数量 |

---

## 📊 输出结果说明

### 主要输出数据

运行 `run_simulation()` 后返回包含以下字段的字典：

```python
results = {
    'temperature': np.ndarray,        # 温度数组
    'energy': np.ndarray,             # 平均能量数组
    'magnetization': np.ndarray,     # 磁化强度数组
    'susceptibility': np.ndarray,     # 磁化率数组
    'specific_heat': np.ndarray,      # 比热数组
    'config': dict,                  # 模拟配置信息
    'timing': dict                   # 性能计时数据
}
```

### 文件输出结构

每次运行会创建带时间戳的结果目录：

```
simulation_results_YYYYMMDD_HHMMSS_XXXX/
├── figures/                    # 结果图表
├── spin_configurations/         # 自旋配置数据
├── raw_simulation_data.npz     # 完整原始数据
├── raw_data_readme.txt        # 数据使用说明
└── results_summary.txt        # 结果汇总
```

---

## 🎯 使用场景示例

### 1. 基础研究模拟

```python
# 小规模快速测试
simulator = xy_sim.XYModelSimulator(
    lattice_size=8,
    equilibrium_steps=500,
    measurement_steps=2000,
    random_seed=123
)

results = simulator.run_simulation(
    temperature_range=(0.5, 2.0),
    num_temperatures=8
)
```

### 2. 高精度研究模拟

```python
# 大规模高精度模拟
simulator = xy_sim.XYModelSimulator(
    lattice_size=32,
    equilibrium_steps=5000,
    measurement_steps=50000,
    random_seed=456
)

results = simulator.run_simulation(
    temperature_range=(0.8, 1.5),
    num_temperatures=15
)
```

### 3. 临界现象研究

```python
# 聚焦相变温度区域的精细扫描
simulator = xy_sim.XYModelSimulator(
    lattice_size=24,
    equilibrium_steps=2000,
    measurement_steps=20000,
    random_seed=789
)

results = simulator.run_simulation(
    temperature_range=(0.85, 1.25),  # XY模型临界温度附近
    num_temperatures=20
)
```

---

## 📈 性能优化特性

### 智能内存管理
- **内存池**: 自动复用数组，减少内存分配开销
- **就地操作**: 尽可能使用就地计算，节省内存
- **垃圾回收**: 智能垃圾回收，防止内存泄漏

### 算法优化
- **动态收敛检测**: 自动检测系统平衡，提前结束热化阶段
- **自适应批处理**: 根据系统规模调整批处理大小
- **智能采样**: 减少样本相关性，提高统计精度
- **早期退出**: 无键情况下的快速路径

### 计算优化
- **向量化操作**: 全面使用NumPy向量化计算
- **三角函数查找表**: 预计算三角函数值，避免重复计算
- **边界条件预计算**: 预先计算周期性边界索引

---

## 🔧 高级功能

### 单温度点运行

```python
# 运行单个温度点
result = simulator._run_temperature_ultra_optimized(temperature=1.0)
print(f"温度: {result['temperature']}")
print(f"能量: {result['energy']:.6f}")
print(f"磁化强度: {result['magnetization']:.6f}")
```

### 获取自旋配置

```python
# 从结果中获取自旋配置
output_dir = simulator.get_output_directory()
spin_data = simulator.load_raw_data()

# 访问特定温度点的自旋配置
spins = spin_data['spin_configurations']  # shape: (n_temps, L, L)
spin_at_T_1_0 = spins[0]  # 第一个温度点的自旋配置
```

### 性能分析

```python
# 获取性能数据
timing = results['timing']
print(f"总耗时: {timing['total_time']:.2f} 秒")
print(f"平均每温度点: {timing['avg_time_per_temp']:.2f} 秒")
print(f"最快温度点: {timing['min_time']:.2f} 秒")
print(f"最慢温度点: {timing['max_time']:.2f} 秒")

# 获取内存池统计
stats = simulator.memory_pool.get_stats()
print(f"内存池命中率: {stats['hit_rate']*100:.1f}%")
```

---

## 📋 数据分析指导

### 物理量分析

```python
import matplotlib.pyplot as plt
import numpy as np

# 绘制能量-温度曲线
plt.figure(figsize=(10, 6))
plt.plot(results['temperature'], results['energy'], 'b-o')
plt.xlabel('温度 (T)')
plt.ylabel('平均能量 (E/N)')
plt.title('XY模型能量-温度关系')
plt.grid(True)
plt.show()

# 绘制磁化强度-温度曲线
plt.figure(figsize=(10, 6))
plt.plot(results['temperature'], results['magnetization'], 'r-s')
plt.xlabel('温度 (T)')
plt.ylabel('磁化强度 (M)')
plt.title('XY模型磁化强度-温度关系')
plt.grid(True)
plt.show()

# 绘制磁化率-温度曲线（识别相变点）
plt.figure(figsize=(10, 6))
plt.plot(results['temperature'], results['susceptibility'], 'g-^')
plt.xlabel('温度 (T)')
plt.ylabel('磁化率 (χ)')
plt.title('XY模型磁化率-温度关系')
plt.grid(True)
plt.show()
```

### 临界温度估算

```python
# 通过磁化率峰值估算临界温度
chi_max_idx = np.argmax(results['susceptibility'])
estimated_Tc = results['temperature'][chi_max_idx]

print(f"估算的临界温度 Tc ≈ {estimated_Tc:.3f}")
print(f"最大磁化率 χ_max = {results['susceptibility'][chi_max_idx]:.3f}")
```

---

## ⚠️ 注意事项

### 计算资源
- **内存需求**: 约为 `8 × L² × L²` 字节
- **计算复杂度**: O(L² × total_steps)
- **推荐配置**: 
  - L ≤ 16: 普通笔记本电脑
  - 16 < L ≤ 32: 工作站级别
  - L > 32: 高性能计算集群

### 参数建议
- **平衡步数**: 建议至少 `500 × L²`
- **测量步数**: 建议至少 `1000 × L²`
- **温度点数**: 相变区域建议 ≥ 15个点
- **随机种子**: 为保证结果可重现，建议设置固定种子

### 数值稳定性
- 温度不要低于 0.01（避免数值溢出）
- 温度不要高于 10.0（物理意义较小）
- 大晶格（L > 64）可能需要更多内存

---

## 🐛 常见问题与解决方案

### Q1: 运行时出现内存不足错误
**解决方案**: 
- 减小晶格尺寸
- 减少测量步数
- 使用更少的温度点

### Q2: 结果不稳定，波动较大
**解决方案**:
- 增加平衡步数
- 增加测量步数
- 检查随机种子设置

### Q3: 运行速度较慢
**解决方案**:
- 确保使用优化的NumPy版本
- 考虑使用较小的晶格进行初步测试
- 调整批处理参数

### Q4: 文件保存失败
**解决方案**:
- 检查磁盘空间
- 确保写入权限
- 检查路径是否有效

---

## 📚 理论背景

### XY模型
XY模型是统计物理学中的重要模型，描述二维平面上的自旋系统：
- 每个自旋可以指向平面上的任意角度 [0, 2π)
- 哈密顿量: H = -J Σ〈i,j〉 cos(θᵢ - θⱼ)
- 具有连续对称性和KT相变

### Swendsen-Wang算法
- 聚类算法，减少临界 slowing down
- 通过构建自旋簇来更新系统
- 在临界点附近效率显著高于传统算法

---

## 📞 技术支持

如遇到问题，请检查：
1. Python版本 ≥ 3.7
2. NumPy版本 ≥ 1.19
3. matplotlib版本 ≥ 3.0（可选，用于可视化）
4. 系统内存是否充足

更多技术细节请参考源码中的详细注释。

---

**版本**: 优化版本 v2.0  
**最后更新**: 2025年12月  
**兼容性**: Python 3.7+, NumPy 1.19+