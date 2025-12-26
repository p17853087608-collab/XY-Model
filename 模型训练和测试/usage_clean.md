# 标准ResNet模型使用指南（优化数据增强版）

## 概述

本项目使用标准ResNet模型进行磁化率预测，专注于图像特征分类，去除了温度感知功能以简化模型结构并提升性能。针对16x16小尺寸灰度图像优化了数据增强策略，采用合适的数据增强技术提升模型性能。

**重要更新**：基于实际测试效果，已禁用Mixup数据增强以获得更好的预测性能。

## 🚀 激进数据增强原理

### 为何使用激进数据增强？

针对16x16小尺寸灰度图像的挑战：
- **信息量有限**: 仅256像素，容易过拟合
- **特征稀疏**: 细节信息相对较少
- **样本不足**: 训练数据量通常有限

### 核心技术优势

#### 1. Cutout（随机遮挡）
- **原理**: 随机遮挡图像区域，强制模型学习全局特征
- **优势**: 减少模型对局部特征的过度依赖
- **适用场景**: 小图像中的关键特征分布较均匀

#### 2. Mixup（样本混合）- 已禁用
- **原理**: 线性混合两张图像和标签
- **原优势**: 线性行为之间产生更平滑的决策边界
- **现状**: 已禁用，因为对预测效果产生负面影响
- **注意**: Mixup仅在训练时使用，预测时从不使用

#### 3. 激进几何变换
- **大幅旋转**: 模拟不同角度观察
- **显著缩放**: 模拟不同距离观察
- **透视变形**: 模拟3D投影变化
- **光照调整**: 模拟不同拍摄条件

## 数据集结构

```
32x32_data_set/
├── train/
│   ├── Amorphous phase/          # 无序相 (3.9-4.0)
│   │   ├── 1295_3.900_spin_config_t_3.900.png
│   │   └── ...
│   └── Ordered phase/           # 有序相 (0.001-0.1)
│       ├── 1012_0.001_spin_config_t_0.001.png
│       └── ...
└── test/
    └── [测试文件夹]
```

## 文件说明

### 核心文件
- `model_clean.py`: 简化的标准ResNet模型定义
- `model_train_clean.py`: 标准训练脚本
- `predict_clean.py`: 标准预测脚本

## 使用方法

### 1. 训练模型

```bash
python model_train_clean.py
```

**训练特性:**
- 使用优化的数据增强策略提升泛化能力
- Cutout遮挡增强模型鲁棒性
- **已禁用Mixup**（避免影响预测效果）
- 温和几何变换：旋转±15°、平移±10%、缩放±10%、透视变换等
- 自适应batch size（NPU/GPU/CPU环境）
- 实时保存最佳模型
- 详细训练数据记录和分析

### 2. 预测

**使用TTA（推荐）:**
```bash
python predict_clean.py
```

**不使用TTA:**
```bash
python predict_clean.py --no-tta
```

**自定义路径:**
```bash
python predict_clean.py --base_path "custom/test/path" --output "custom_output"
```

## 模型架构

### ResNet1
- **输入**: 1×224×224 灰度图像
- **骨干网络**: ResNet结构（4个残差块）
- **分类层**: 全连接层输出2个类别
- **输出**: [非晶相概率, 有序相概率]

### 优化数据增强策略（训练时）
- **基础几何变换**:
  - 水平翻转: p=0.5
  - 垂直翻转: p=0.2（新增）
  - 随机旋转: ±15度（温和增强）
  - 随机仿射变换: 旋转±10度、平移±10%、缩放±10%、剪切±5度（温和增强）
  - 透视变换: p=0.3, distortion_scale=0.1
- **颜色/光照变换**:
  - 亮度调整: brightness=0.3（新增）
  - 对比度调整: contrast=0.3（新增）
- **高级增强技术**:
  - **Cutout遮挡**: 1个16x16像素遮挡块
  - **Mixup混合**: 已禁用（影响预测效果）
- **标准化**: mean=[0.5], std=[0.5]

### TTA（测试时增强）
共10种变换策略:
- 原始图像
- 水平翻转
- 垂直翻转
- ±15度旋转
- ±15%缩放
- 10%平移变换（新增）
- 透视变换（新增）
- 亮度/对比度调整（新增）

## 输出文件

### 训练输出
- `best.pth`: 最佳模型权重
- `training_curves.pdf`: 训练曲线图

### 数据保存功能（增强版）
在`数据保存`文件夹中自动生成：
- `detailed_training_data_clean.csv`: 每个样本的详细损失数据（包含Mixup信息）
- `loss_stats_clean.json`: 损失函数统计信息
- `image_features_sparse_clean.npz`: 图像特征稀疏矩阵
- `feature_metadata_clean.csv`: 特征对应的标签信息
- `feature_statistics_clean.json`: 特征统计和稀疏度信息

**Mixup数据追踪（已禁用）**:
- `mixup_applied`: 是否应用Mixup
- `mixup_lambda`: Mixup混合系数

### 预测输出
- `save_data/{folder_name}.csv`: 详细预测结果
- `save_data/{folder_name}_stats.csv`: 文件夹统计信息
- `save_data/overall_summary.csv`: 总体统计

## 数据分析工具

### 使用data_analyzer_clean.py
训练完成后运行分析脚本：
```bash
python data_analyzer_clean.py
```

### 分析功能
1. **数据加载**: 加载保存的稀疏矩阵和训练数据
2. **分布分析**: 标签分布、稀疏度、特征统计
3. **损失分析**: 训练vs验证对比、准确率变化、损失分布
4. **PCA降维**: 2D/3D可视化、解释方差分析
5. **t-SNE分析**: 非线性降维可视化
6. **聚类分析**: K-means聚类，分析类别分布
7. **可视化图表**:
   - `loss_analysis_clean.png`: 损失函数分析图
   - `pca_analysis_clean.png`: PCA降维可视化
   - `tsne_analysis_clean.png`: t-SNE非线性降维
   - `cluster_analysis_clean.png`: 聚类结果可视化

### 分析报告
- `feature_analysis_report_clean.json`: 完整的分析报告，包含所有统计信息

### 预测结果格式
```csv
image_id,class,prediction,ordered_probability,amorphous_probability
1295_3.900,Ordered phase,1,0.998523,0.001477
```

## 性能优化

### 硬件适配
- **昇腾NPU**: batch_size=64, num_workers=0
- **GPU**: batch_size=32, num_workers=4  
- **CPU**: batch_size=32, num_workers=2

### 训练参数
- 学习率: 0.001
- 优化器: Adam
- 损失函数: CrossEntropyLoss（支持Mixup混合损失）
- 训练轮数: 20（可调整）
- Mixup Alpha: 1.0（可调整）
- Cutout遮挡: 32x32像素（可调整）

## 高级配置

### 自定义数据增强参数
可在`model_train_clean.py`中调整：
```python
# 主函数中的配置
use_mixup = False         # 已禁用Mixup（可设为True重新启用）
mixup_alpha = 0.4         # Mixup强度（如需启用）

# 训练函数中的配置
def train_model_process(model, train_loader, val_loader, num_epochs, use_mixup=False, mixup_alpha=0.4):

# Cutout参数调整（在Cutout类中）
Cutout(n_holes=1, length=16)  # 调整遮挡数量和大小
```

### 数据增强强度对比
| 参数 | 保守策略 | 优化策略（当前） |
|------|----------|------------------|
| 旋转角度 | ±10° | ±30° |
| 平移范围 | ±5% | ±15% |
| 缩放范围 | 0.95-1.05 | 0.8-1.2 |
| TTA变换数 | 6种 | 10种 |

## 故障排除

### 常见问题
1. **模型加载失败**: 检查`best.pth`文件是否存在
2. **内存不足**: 减少batch_size或禁用部分数据增强
3. **预测错误**: 确认图像路径正确
4. **设备检测失败**: 检查PyTorch和驱动安装
5. **训练过慢**: 减少TTA变换数量或降低数据增强强度
6. **过拟合**: 增加数据增强强度或使用更激进的参数

### 调试建议
- 检查数据集路径格式
- 确认PyTorch版本兼容性
- 验证图像文件完整性
- 监控GPU内存使用

## 支持的图像格式
- PNG
- JPG/JPEG
- BMP
- GIF

## 环境要求
- Python 3.7+
- PyTorch 1.9.0+
- torchvision
- PIL
- numpy
- pandas
- matplotlib
- tqdm (可选，用于进度条)

---

**重要更新**: 
- **已禁用Mixup数据增强**：基于实际测试，Mixup会显著降低预测效果，已从训练中移除
- **预测时从不使用Mixup**：Mixup仅用于训练，预测使用TTA（测试时增强）
- **数据增强强度优化**：调整为更温和的参数以获得更好的预测性能

**注意**: 
- 如果需要恢复温度感知功能，请使用原版本的 `model.py`, `model_train.py`, 和 `predict.py` 文件。
- 如需重新启用Mixup，可在`model_train_clean.py`中将`use_mixup = False`改为`True`。
- 当前使用优化的数据增强策略，平衡泛化能力和预测性能。