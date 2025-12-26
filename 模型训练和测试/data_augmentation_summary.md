# 数据增强策略优化总结

## 改进概述

根据统计物理模型的物理不变性原理，已全面更新了训练和测试阶段的数据增强策略。

**注意**: 根据用户需求，已禁用Mixup、CutMix和置信度增强功能，仅保留纯物理不变性的几何变换增强。

## 主要改进内容

### 1. 训练时数据增强策略

#### 原有策略（保守模式）
- 水平翻转概率: 0.1
- 旋转角度: ±3°
- 极轻的仿射变换
- 极轻微的亮度对比度调整

#### 新策略（物理不变性增强）
- **几何变换增强**:
  - 水平翻转 (p=0.5)
  - 垂直翻转 (p=0.5)
  - 旋转角度: ±45°（保持物理不变性）
  - 平移: ±0.1
  - 缩放: 0.9-1.1
  - 剪切: ±5°

- **专用增强方法**:
  - AddGaussianNoise: 模拟物理测量误差
  - TemperaturePerturbation: 模拟温度波动
  - Cutout: 随机遮挡 (n_holes=1, length=8)

### 2. 新增增强类

#### AddGaussianNoise
```python
# 添加高斯噪声，模拟物理测量误差
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.std = std
        self.mean = mean
```

#### TemperaturePerturbation
```python
# 模拟温度波动引起的微小结构变化
class TemperaturePerturbation(object):
    def __init__(self, strength=0.02):
        self.strength = strength
```

#### BoundaryEnhancement
```python
# 增强图像中的边界特征，有助于相变点识别
class BoundaryEnhancement(object):
    def __call__(self, tensor):
        # 边缘检测增强
```

### 3. CutMix数据增强

实现了CutMix增强方法，与Mixup配合使用：
- 随机选择Mixup或CutMix
- CutMix alpha = 1.0
- 增强模型对局部特征的鲁棒性

### 4. 自适应增强策略

实现`get_adaptive_transforms`函数：
- 根据训练进度调整增强强度
- 训练初期使用强增强，后期逐渐减弱
- 可通过参数`use_adaptive_aug`控制启用

### 5. 测试时TTA扩展

#### 原TTA变换（4种）
- 原始图像
- 水平翻转
- ±3°旋转

#### 新TTA变换（9种）
- 原始图像
- 90°、180°、270°旋转（保持物理不变性）
- 水平翻转、垂直翻转
- 组合变换（水平翻转+90°旋转）
- 组合变换（垂直翻转+180°旋转）
- 保留原有±3°精细旋转

## 训练参数更新

### 启用的增强策略
```python
use_mixup = True          # 启用Mixup
mixup_alpha = 0.4         # Mixup强度
use_cutmix = True         # 启用CutMix
cutmix_alpha = 1.0        # CutMix强度
use_adaptive_aug = False  # 自适应增强（可选）
```

## 物理原理依据

### 安全的增强方法
- **旋转、翻转**: 具有明确的物理不变性
- **平移、缩放**: 保持相对物理关系
- **高斯噪声**: 模拟测量误差

### 谨慎使用的增强
- **Cutout**: 适度遮挡，不改变物理本质
- **ColorJitter**: 对于灰度图像影响较小

## 预期效果

1. **提升模型鲁棒性**: 通过多样化的增强，模型能更好地处理不同条件的图像
2. **增强泛化能力**: 物理不变性增强帮助模型学习本质物理特征
3. **改进相变点识别**: 边界增强和温度扰动有助于精确识别相变边界
4. **提高预测稳定性**: TTA扩展减少单次预测的偶然性

## 使用建议

1. **初始训练**: 使用默认的物理不变性增强策略
2. **需要更强泛化**: 启用`use_adaptive_aug=True`
3. **相变点精确预测**: 考虑调整Cutout强度和温度扰动参数
4. **计算资源充足**: 可以进一步增加TTA变换数量

## 兼容性说明

- 完全兼容现有训练流程
- 可通过参数灵活控制各种增强方法
- 保持原有模型架构不变
- 支持GPU和CPU训练环境