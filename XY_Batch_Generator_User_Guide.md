# XY模型批量自旋图生成器使用指南

## 1. 功能概述

**XY模型批量自旋图生成器**是一个高度可配置的工具，用于批量生成不同温度下的XY模型自旋配置图像。主要特点包括：

- **模块化设计**：核心功能封装为`XYBatchGenerator`类，接口清晰
- **灵活配置**：支持自定义温度点、晶格尺寸、图像数量等参数
- **并行加速**：多进程并行处理，支持GPU加速
- **自动分类**：按温度自动分类存储图像（低温/中温/高温）
- **任务报告**：自动生成详细的生成统计报告

## 2. 安装准备

### 2.1 环境要求
- Python 3.8+
- 依赖包: `numpy matplotlib tqdm`
- 可选依赖: `cupy`（用于GPU加速）

### 2.2 安装命令
```bash
# 创建虚拟环境（推荐）
conda create -n xy_gen python=3.9 -y
conda activate xy_gen

# 安装依赖
pip install numpy matplotlib tqdm

# 如需GPU加速（根据CUDA版本选择）
# pip install cupy-cuda11x
```

### 2.3 文件准备
确保以下文件在同一目录：
- `enhanced_batch_generator.py`（封装后脚本）
- `xy_model_simulator.py`（XY模型模拟器）

## 3. 快速入门

### 3.1 基础使用（命令行模式）
```bash
# 默认参数运行（生成小规模测试集）
python enhanced_batch_generator.py

# 自定义参数运行
python enhanced_batch_generator.py --lattice 128 --images 100 --gpu --temps 0.5 1.0 1.5 2.0
```

### 3.2 代码调用模式
```python
from enhanced_batch_generator import XYBatchGenerator, GeneratorConfig

# 创建配置
config = GeneratorConfig()
config.lattice_size = 64          # 晶格尺寸64x64
config.images_per_temp = 50       # 每个温度点生成50张图像
config.temp_points = [0.3, 0.7, 1.0, 1.5]  # 温度点列表
config.use_gpu = True             # 启用GPU加速

# 创建生成器并运行
generator = XYBatchGenerator(config)
generator.set_output_dir("my_custom_output")  # 设置自定义输出目录
report = generator.generate()

print(f"生成完成！输出目录: {report['output_dir']}")
print(f"成功生成图像: {report['success_images']}张")
```

## 4. 核心接口说明

### 4.1 `GeneratorConfig` 配置类
| 参数名称               | 类型   | 默认值              | 说明                          |
|------------------------|--------|---------------------|-------------------------------|
| `lattice_size`         | int    | 64                  | 晶格尺寸（4-256）              |
| `temp_points`          | list   | [0.3, 0.7, 1.0, 1.3, 1.7, 2.2] | 温度点列表           |
| `images_per_temp`      | int    | 100                 | 每个温度点生成图像数量         |
| `use_gpu`              | bool   | False               | 是否启用GPU加速                |
| `parallel_workers`     | int    | None                | 并行进程数（None自动适配）     |
| `output_base`          | str    | "batch_output"      | 基础输出目录                  |
| `dpi`                  | int    | 150                 | 图像分辨率                    |
| `equilibrium_steps`    | int    | 1000                | 平衡步数                      |
| `measurement_steps`    | int    | 2000                | 测量步数                      |

### 4.2 `XYBatchGenerator` 核心类方法
| 方法名称               | 参数                 | 说明                          |
|------------------------|----------------------|-------------------------------|
| `__init__(config)`     | config: 配置对象     | 初始化生成器                  |
| `set_output_dir(path)` | path: 自定义路径     | 设置自定义输出目录            |
| `generate()`           | 无                  | 执行批量生成任务              |

## 5. 高级配置示例

### 5.1 自定义温度点和输出目录
```python
# 创建配置
config = GeneratorConfig() 
config.temp_points = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]  # 密集温度点
config.output_base = "phase_transition_study"  # 相变研究专用输出目录

# 创建生成器
generator = XYBatchGenerator(config)
generator.set_output_dir("D:/xy_model_results/20251129")  # 绝对路径
generator.generate()
```

### 5.2 大规模生成配置
```python
config = GeneratorConfig()
config.lattice_size = 256        # 大晶格尺寸
config.images_per_temp = 200     # 每个温度点200张图像
config.parallel_workers = 8      # 强制8进程并行
config.use_gpu = True            # GPU加速
config.equilibrium_steps = 2000  # 增加平衡步数确保收敛

generator = XYBatchGenerator(config)
generator.generate()
```

## 6. 命令行参数说明

| 参数         | 缩写 | 说明                          | 示例                          |
|--------------|------|-------------------------------|-------------------------------|
| `--lattice`  | -l   | 晶格尺寸                      | `--lattice 128`               |
| `--images`   | -n   | 每个温度点图像数量            | `--images 150`                |
| `--gpu`      | -g   | 启用GPU加速                   | `--gpu`                       |
| `--output`   | -o   | 自定义输出目录                | `--output ./results`           |
| `--temps`    | -t   | 自定义温度点列表              | `--temps 0.5 1.0 1.5 2.0`     |

## 7. 输出目录结构

```
batch_output/                     # 基础输出目录
└── batch_20251129_153022/        # 带时间戳的批次目录
    ├── images/                   # 图像存储目录
    │   ├── low_temp/             # 低温图像 (T<0.5K)
    │   ├── medium_temp/          # 中温图像 (0.5K≤T<1.5K)
    │   └── high_temp/            # 高温图像 (T≥1.5K)
    └── batch_report.json         # 批次报告（JSON格式）
```

## 8. 常见问题解决

### 8.1 模拟器导入失败
**错误提示**：`ModuleNotFoundError: No module named 'xy_model_simulator'`  
**解决**：确保`xy_model_simulator.py`与生成器脚本在同一目录

### 8.2 GPU加速无效
**错误提示**：`警告: CuPy未安装，将使用CPU模式`  
**解决**：安装对应CUDA版本的CuPy：`pip install cupy-cuda11x`（根据CUDA版本调整）

### 8.3 内存溢出
**症状**：进程意外退出或提示内存不足  
**解决**：减小晶格尺寸（如从256→128）、降低并行进程数或减少单温度图像数量

### 8.4 中文乱码
**症状**：图像标题或标签中文显示乱码  
**解决**：确保matplotlib字体配置正确，或在生成图像时指定字体：
```python
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
```