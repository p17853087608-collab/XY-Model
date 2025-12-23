# XY模型模拟器快速参考

## 🚀 一行运行
```python
import xy_model_simulator_optimized as xy_sim
sim = xy_sim.XYModelSimulator(lattice_size=16, equilibrium_steps=1000, measurement_steps=10000, random_seed=42)
results = sim.run_simulation(temperature_range=(0.5, 2.5), num_temperatures=10)
```

## 📊 结果访问
```python
# 物理量
temperatures = results['temperature']
energies = results['energy']
magnetizations = results['magnetization']
susceptibilities = results['susceptibility']
specific_heats = results['specific_heat']

# 性能数据
timing = results['timing']
total_time = timing['total_time']
output_dir = sim.get_output_directory()
```

## ⚙️ 常用参数组合

| 场景 | 晶格 | 平衡步 | 测量步 | 用途 |
|------|------|--------|--------|------|
| 快速测试 | L=8 | 500 | 2000 | 代码调试 |
| 标准研究 | L=16 | 1000 | 10000 | 常规模拟 |
| 高精度 | L=32 | 5000 | 50000 | 精确研究 |
| 临界研究 | L=24 | 2000 | 20000 | 相变分析 |

## 🔧 优化建议
- **小晶格(L≤8)**: 使用快速测试参数
- **中晶格(16≤L≤32)**: 标准参数即可
- **大晶格(L>32)**: 需要大内存，考虑HPC
- **相变区**: 增加温度点密度(≥15个)

## 📁 输出文件结构
```
simulation_results_YYYYMMDD_HHMMSS_XXXX/
├── spin_configurations/     # 自旋配置 .npy 文件
├── figures/                # 图表 .png 文件  
├── raw_simulation_data.npz  # 完整数据
└── raw_data_readme.txt     # 数据说明
```

## ⚠️ 注意事项
- 内存需求 ≈ 8 × L⁴ 字节
- 温度范围: 0.01 < T < 10.0
- 设置random_seed保证可重现性
- 大规模运行建议HPC环境

---
**完整文档**: `XY模型模拟器操作文档.md`