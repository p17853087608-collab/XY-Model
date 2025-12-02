# XY模型模拟器操作文档（CPU版本）

## 快速开始

### 1. 导入模块
```python
from xy_model_simulator_optimized import XYModelSimulator
```

### 2. 创建模拟器
```python
# 基本用法（使用默认参数）
simulator = XYModelSimulator()

# 自定义参数
simulator = XYModelSimulator(
    lattice_size=16,          # 晶格尺寸 16×16
    equilibrium_steps=1000,   # 平衡步数
    measurement_steps=10000,  # 测量步数
    interaction_constant=1.0, # 相互作用常数J
    random_seed=42,          # 随机种子
    use_gpu=False           # 强制使用CPU（默认）
)
```

### 3. 运行模拟
```python
# 使用默认温度范围 (0.1, 2.5)，10个温度点
results = simulator.run_simulation()

# 自定义温度范围和温度点数量
results = simulator.run_simulation(
    temperature_range=(0.5, 2.0),  # 温度范围
    num_temperatures=15            # 温度点数量
)
```

### 4. 获取结果
```python
# 获取完整的模拟结果字典
results = simulator.get_results()

# 获取配置信息
config = simulator.get_config()

# 获取输出目录路径
output_dir = simulator.get_output_directory()
```

## 参数说明

### 初始化参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lattice_size` | 16 | 晶格尺寸(L×L)，建议8-64 |
| `equilibrium_steps` | 1000 | 系统平衡步数，建议500-5000 |
| `measurement_steps` | 10000 | 物理量测量步数，建议5000-50000 |
| `interaction_constant` | 1.0 | 交换相互作用常数J |
| `random_seed` | None | 随机种子，用于重现结果 |
| `use_gpu` | False | 是否使用GPU（CPU版本固定为False） |

### 运行参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `temperature_range` | (0.1, 2.5) | 温度范围(kBT/J) |
| `num_temperatures` | 10 | 温度点数量 |

## 结果输出

### 自动生成的文件
运行完成后会自动生成以下文件和文件夹：

```
simulation_results_YYYYMMDD_HHMMSS/
├── figures/                          # 图表文件夹
│   ├── energy_vs_temperature_optimized.pdf
│   ├── specific_heat_vs_temperature_optimized.pdf
│   ├── magnetization_vs_temperature_optimized.pdf
│   └── susceptibility_vs_temperature_optimized.pdf
├── spin_configurations/              # 自旋配置图片
│   ├── spin_config_T_0.100_optimized.png
│   ├── spin_config_T_0.364_optimized.png
│   └── ...
├── simulation_results_optimized.txt  # 数值结果
└── simulation_config_optimized.txt   # 配置信息
```

### 结果数据结构
```python
results = {
    'temperature': array([...]),        # 温度数组
    'energy': array([...]),              # 能量数组
    'magnetization': array([...]),       # 磁化强度数组
    'specific_heat': array([...]),       # 比热数组
    'susceptibility': array([...]),      # 磁化率数组
    'config': {...},                     # 配置信息
    'timing': {...}                      # 性能统计
}
```

## 实用示例

### 示例1：快速测试
```python
# 小规模快速测试
simulator = XYModelSimulator(
    lattice_size=8,
    equilibrium_steps=200,
    measurement_steps=500,
    random_seed=42
)
results = simulator.run_simulation(num_temperatures=5)
print("测试完成，结果保存在:", simulator.get_output_directory())
```

### 示例2：正式模拟
```python
# 正式研究用模拟
simulator = XYModelSimulator(
    lattice_size=32,
    equilibrium_steps=2000,
    measurement_steps=20000,
    interaction_constant=1.0,
    random_seed=12345
)

# 在临界温度附近进行精细扫描
results = simulator.run_simulation(
    temperature_range=(0.8, 1.2),
    num_temperatures=20
)

# 获取并分析结果
temp = results['temperature']
energy = results['energy']
mag = results['magnetization']

# 找到比热峰位置（相变温度）
specific_heat = results['specific_heat']
tc_index = specific_heat.argmax()
print(f"相变温度大约在: {temp[tc_index]:.3f}")
```

### 示例3：批量模拟
```python
# 对不同晶格尺寸进行批量模拟
sizes = [8, 16, 24, 32]
all_results = {}

for size in sizes:
    print(f"正在模拟晶格尺寸 {size}×{size}...")
    simulator = XYModelSimulator(
        lattice_size=size,
        equilibrium_steps=1000,
        measurement_steps=10000,
        random_seed=42
    )
    results = simulator.run_simulation(num_temperatures=15)
    all_results[size] = results
    print(f"完成，结果保存在: {simulator.get_output_directory()}")
```

## 性能说明

### CPU优化特性
- **内存池管理**：减少动态内存分配
- **并行温度点处理**：多线程并行计算
- **向量化操作**：使用NumPy优化计算
- **算法优化**：高效的并查集实现
- **缓存机制**：缓存三角函数计算结果

### 性能建议
1. **晶格尺寸**：建议8-64，过大可能很慢
2. **步数设置**：平衡质量和速度的权衡
3. **温度点数量**：10-20个通常足够
4. **多核利用**：程序会自动使用多核CPU

## 常见问题

### Q: 如何重现结果？
A: 设置相同的`random_seed`即可重现结果。

### Q: 程序运行很慢怎么办？
A: 
- 减小`lattice_size`
- 减少`equilibrium_steps`和`measurement_steps`
- 减少`num_temperatures`

### Q: 如何更改输出文件格式？
A: 修改`plot_results()`调用：
```python
simulator.plot_results(file_format='png')  # 可选：pdf, png, svg
```

### Q: 如何获取中间结果？
A: 直接调用`_run_temperature_optimized()`方法：
```python
single_result = simulator._run_temperature_optimized(temperature=1.0)
```

## 注意事项

1. **纯CPU计算**：此版本不使用GPU，适合无GPU环境
2. **内存需求**：晶格尺寸不宜过大，避免内存不足
3. **时间消耗**：大规模模拟可能需要较长时间
4. **结果保存**：程序会自动保存所有结果，无需手动操作

## 输出解读

### 物理量说明
- **Energy (能量)**：系统总能量，反映系统状态
- **Magnetization (磁化强度)**：磁有序程度，低温时较高
- **Specific Heat (比热)**：能量涨落，在相变点出现峰
- **Susceptibility (磁化率)**：磁化涨落，在相变点出现峰

### 相变识别
- 比热和磁化率的峰值位置对应Kosterlitz-Thouless相变
- 2D XY模型的相变温度约为T ≈ 0.89J/kB

---

*此文档涵盖CPU版本XY模型模拟器的完整使用方法，如需更多技术细节请参考源代码注释。*

根据您提供的信息，问题在于您的系统GLIBC版本过低（当前为2.17），而Miniconda的最新安装程序要求GLIBC >=2.28。这导致安装失败。您的系统可能是CentOS 7或类似的老版本Linux，其GLIBC 2.17无法支持新版本的Miniconda。

### 解决方案：
下载并安装一个与GLIBC 2.17兼容的旧版本Miniconda。通常，Miniconda for Python 3.7或更早的版本支持GLIBC 2.17。以下是具体步骤：

1. **删除当前安装失败的文件**（如果存在）：
   ```bash
   rm -rf ~/miniconda3
   rm Miniconda3-latest-Linux-x86_64.sh
   ```

2. **下载旧版本Miniconda**：例如，Miniconda3 for Python 3.7（版本4.8.3），该版本已知支持GLIBC 2.17。使用wget下载：
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
   ```

3. **安装Miniconda**：
   ```bash
   chmod +x Miniconda3-py37_4.8.3-Linux-x86_64.sh
   bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -f -p ~/miniconda3
   ```
   - `-b`：批量模式，自动接受许可证。
   - `-f`：强制安装，覆盖现有目录。
   - `-p ~/miniconda3`：指定安装路径。

4. **激活Conda环境**：
   ```bash
   source ~/miniconda3/bin/activate
   ```
   激活后，提示符应该显示`(base)`，表示Conda基础环境已激活。

5. **验证安装**：
   ```bash
   conda --version
   ```
   应该输出Conda版本号（如4.8.3）。

### 注意事项：
- **Python版本**：此安装将使用Python 3.7。如果您需要其他Python版本，安装后可以使用`conda install python=3.8`等命令升级，但请确保新版本仍支持您的系统。
- **后续使用**：为了永久激活Conda，可以将`source ~/miniconda3/bin/activate`添加到您的`~/.bashrc`文件中：
  ```bash
  echo "source ~/miniconda3/bin/activate" >> ~/.bashrc
  source ~/.bashrc
  ```
- **其他版本**：如果上述版本不工作，您可以尝试其他旧版本。查看Miniconda存档页面：https://repo.anaconda.com/miniconda/，选择文件名中包含`py37`或更早版本（如`py36`）的安装程序。

### 如果问题依旧：
如果安装后仍有问题，可能是系统环境问题。请检查：
- 确保安装路径正确：`ls ~/miniconda3/bin/activate`应该存在。
- 如果GLIBC版本仍然不兼容，考虑使用更旧的Miniconda版本（如Python 3.6版本）。

如果您需要进一步协助，请提供您的系统详细信息（如`cat /etc/redhat-release`或`uname -a`）。