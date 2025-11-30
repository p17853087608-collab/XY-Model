# XY模型模拟结果绘图工具使用文档

## 1. 概述

`xy_plotter.py`是一个独立的绘图工具，用于可视化XY模型蒙特卡洛模拟结果，展示温度与各物理量（能量、磁化强度、比热和磁化率）的关系。

## 2. 安装要求

- Python 3.6+
- 依赖库: numpy, matplotlib

可通过以下命令安装依赖:
```bash
pip install numpy matplotlib
```

## 3. 使用方法

### 3.1 基本用法

从命令行直接运行，指定模拟结果文件路径:

```bash
python xy_plotter.py -i simulation_results.txt
```

### 3.2 命令行参数

| 参数 | 全称 | 描述 | 必需 |
|------|------|------|------|
| -i | --input | 模拟结果文本文件路径 | 是 |
| -f | --format | 输出图像格式，支持pdf/png/svg，默认pdf | 否 |

### 3.3 示例

```bash
# 生成PDF格式图像
python xy_plotter.py -i simulation_results.txt -f pdf

# 生成PNG格式图像
python xy_plotter.py -i simulation_results.txt -f png
```

## 4. 编程接口使用

### 4.1 从文件加载数据并绘图

```python
from xy_plotter import XYPlotter

# 创建绘图器实例
plotter = XYPlotter()

# 加载结果文件
if plotter.load_results("simulation_results.txt"):
    # 绘制所有物理量图像
    plotter.plot_all(file_format='pdf')
    
    # 或单独绘制某个物理量
    # plotter.plot_energy()
    # plotter.plot_magnetization()
    # plotter.plot_specific_heat()
    # plotter.plot_susceptibility()
```

### 4.2 直接传入数据绘图

```python
from xy_plotter import XYPlotter
import numpy as np

# 准备结果数据
results = {
    'temperature': np.linspace(0.1, 2.5, 10),
    'energy': np.array([...]),  # 替换为实际数据
    'magnetization': np.array([...]),  # 替换为实际数据
    'specific_heat': np.array([...]),  # 替换为实际数据
    'susceptibility': np.array([...])  # 替换为实际数据
}

# 创建绘图器并传入数据
plotter = XYPlotter(results)
plotter.plot_all(file_format='png')
```

## 5. 输出说明

### 5.1 输出目录

程序会自动创建`figures`目录，并将生成的图像文件保存在该目录下。

### 5.2 生成的图像文件

| 文件名 | 描述 |
|--------|------|
| energy_vs_temperature | 能量随温度变化图 |
| magnetization_vs_temperature | 磁化强度随温度变化图 |
| specific_heat_vs_temperature | 比热随温度变化图 |
| susceptibility_vs_temperature | 磁化率随温度变化图 |

## 6. 故障排除

### 6.1 常见错误

1. **文件加载失败**
   - 确保输入文件路径正确
   - 检查文件格式是否符合要求（制表符分隔的文本文件）

2. **图像不显示中文**
   - 确保matplotlib已正确配置中文字体
   - 可在脚本开头添加中文字体配置代码

### 6.2 联系支持

如遇到其他问题，请检查系统日志或联系技术支持。