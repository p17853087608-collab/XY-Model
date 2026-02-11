import os
import copy
import time
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model_clean import ResNet1
from model_clean import Residual
import torch
import pandas as pd
import numpy as np
from scipy import sparse
import json

import torch.nn.functional as F



def get_adaptive_transforms(epoch, total_epochs):
    """根据训练进度自适应调整增强强度（简化版本）"""
    strength = max(0.1, 1.0 - epoch / total_epochs)
    
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        
        # 基础几何变换
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=int(30*strength)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05*strength, 0.05*strength),
            scale=(1-0.05*strength, 1+0.05*strength),
            shear=int(3*strength)
        ),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    return train_transform

def get_data_path(relative_path):
    """获取跨平台的正确数据路径"""
    return os.path.join(os.getcwd(), relative_path.replace('\\', os.sep))

# 检查PyTorch版本兼容性
if torch.__version__ < '1.9.0':
    print(f"警告：检测到PyTorch版本 {torch.__version__}，较老版本可能存在兼容性问题")

# 数据处理函数，划分训练集和验证集
def train_val_data_process():
    ROOT_PATH = get_data_path('16x16_data_set\\train')
    
    # 训练集数据增强（简化版本）
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        
        # 基础几何变换
        transforms.RandomHorizontalFlip(p=0.5),           # 水平翻转
        transforms.RandomVerticalFlip(p=0.5),             # 垂直翻转
        transforms.RandomRotation(degrees=30),            # 减小旋转角度
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.05, 0.05),                       # 减小平移范围
            scale=(0.95, 1.05),                          # 减小缩放范围
            shear=3                                      # 减小剪切角度
        ),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    # 验证集不使用数据增强，只做必要的预处理
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # 加载数据集
    full_dataset = ImageFolder(root=ROOT_PATH)
    
    print(f"数据集信息:")
    print(f"  - 样本总数: {len(full_dataset)}")
    print(f"  - 类别数量: {len(full_dataset.classes)}")
    print(f"  - 类别名称: {full_dataset.classes}")
    
    # 划分训练集和验证集，比例为8:2
    train_size = int(len(full_dataset) * 0.8)
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )
    
    # 创建带索引的子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # 为子数据集设置不同的变换
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # 训练集和验证集的数据加载器 
    if hasattr(torch, 'npu'):
        print("昇腾NPU环境，检测到大显存，使用优化batch size")
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers=0)
    elif torch.cuda.is_available():
        print("GPU环境")
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=4)
    else:
        print("CPU环境")
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=2)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"数据集划分完成：训练集 {len(train_dataset)} 张图片，验证集 {len(val_dataset)} 张图片")
    return train_loader, val_loader

# 训练模型的函数
def train_model_process(model, train_loader, val_loader, num_epochs, use_mixup=False, mixup_alpha=0.4, 
                        use_cutmix=False, cutmix_alpha=1.0, use_adaptive_aug=False):
    # 检测昇腾NPU环境
    if hasattr(torch, 'npu'):
        device = torch.device("npu:0")
        print("检测到昇腾NPU，使用NPU进行训练")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("检测到GPU，使用GPU进行训练")
    else:
        device = torch.device("cpu")
        print("使用CPU进行训练")
    
    class_weights = torch.tensor([1.05, 1.0])  
    if device.type == 'cuda':
        class_weights = class_weights.cuda()
    elif device.type == 'npu':
        class_weights = class_weights.npu()
    
    # 标准加权交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    print("✓ 使用标准加权交叉熵损失函数")
    
    # 优化器使用Adam优化器，极低学习率以获得更精细的学习
    # 使用更小学习率进行精细调整，确保准确学习1.15相变点
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)  # 更小学习率进行精细调整
    
    # 余弦退火学习率调度器，帮助优化收敛过程
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=5e-6
    )
    print("✓ 启用余弦退火学习率调度器，优化收敛过程")
    # 将模型发送到所用设备
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 最高准确度
    best_acc = 0.0

    # 训练集损失值列表和验证集损失值列表
    train_loss_list = []
    val_loss_list = []
    # 训练集精度值列表和验证集精度值列表
    train_acc_list = []
    val_acc_list = []
    # AUC指标列表和学习率列表
    auc_list = []
    lr_list = []
    
    # 详细训练数据收集
    detailed_training_data = []

    # 记录训练开始时间
    since = time.time()

    # 打印数据增强配置信息
    print(f"数据增强配置（简化版本）:")
    print(f"  - 几何变换: 水平翻转p=0.5, 垂直翻转p=0.5, 旋转±30°")
    print(f"  - 仿射变换: 平移±5%, 缩放0.95-1.05, 剪切±3°")
    print(f"  - 移除功能: 高斯噪声、温度扰动、Cutout、Mixup、CutMix")
    print(f"  - 类别权重: 有序相1.4, 非晶相1.0")
    print(f"  - 学习率: 0.0001 (精细学习)")
    print(f"  - 权重衰减: 5e-4 (正则化)")

    # 开始训练
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*10)

        # 初始化参数
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 动态调整数据增强策略
        if use_adaptive_aug:
            train_dataset.dataset.transform = get_adaptive_transforms(epoch, num_epochs)
        
        # 初始化AUC相关变量
        all_val_probs = []
        all_val_labels = []
        
        # 初始化epoch数据字典
        epoch_data = {}

        model.train()
        for step,(x,y) in enumerate(train_loader):
            # 特征
            x = x.to(device)
            # 标签
            y = y.to(device)
            
            # 标准训练（不使用Mixup/CutMix）
            outputs = model(x)
            loss = criterion(outputs, y)
            pre_lab = torch.argmax(outputs, dim=1)
            train_corrects += torch.sum(pre_lab == y.data)
            
            # 保存详细训练数据
            for i in range(x.size(0)):
                detailed_training_data.append({
                    'epoch': epoch + 1,
                    'batch': step + 1,
                    'sample': i + 1,
                    'phase': 'train',
                    'loss': loss.item(),
                    'predicted': pre_lab[i].item(),
                    'actual': y[i].item(),
                    'correct': (pre_lab[i] == y[i]).item(),
                    'mixup_applied': use_mixup,
                    'mixup_lambda': lam if use_mixup else 1.0
                })

            # 梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算训练集的损失值和精度值
            train_loss += loss.item() * x.size(0)
            train_num += x.size(0)

        model.eval()
        for step,(x,y) in enumerate(val_loader):
            # 特征
            x = x.to(device)
            # 标签
            y = y.to(device)
            
            with torch.no_grad():
                # 前向传播
                outputs = model(x)
                pre_lab = torch.argmax(outputs, dim=1)
                
                # 计算softmax概率用于AUC
                probs = torch.softmax(outputs, dim=1)
                all_val_probs.extend(probs[:, 1].cpu().numpy())  # 正类概率
                all_val_labels.extend(y.cpu().numpy())

                loss = criterion(outputs, y)
                
                # 保存详细验证数据
                for i in range(x.size(0)):
                    detailed_training_data.append({
                        'epoch': epoch + 1,
                        'batch': step + 1,
                        'sample': i + 1,
                        'phase': 'val',
                        'loss': loss.item(),
                        'predicted': pre_lab[i].item(),
                        'actual': y[i].item(),
                        'correct': (pre_lab[i] == y[i]).item(),
                        'predicted_prob': probs[i, pre_lab[i]].item()
                    })

                # 计算验证集的损失值和精度值
                val_loss += loss.item() * x.size(0)
                val_corrects += torch.sum(pre_lab == y.data)
                val_num += x.size(0)

        # 计算训练集和验证集的loss值
        train_loss_list.append(train_loss/train_num)
        # 对于Mixup训练，使用一个替代的准确率计算方法
        if use_mixup and isinstance(train_corrects, int):
            # 如果使用Mixup且train_corrects是int，使用平均准确率估计
            train_acc = 0.5  # 默认值，Mixup时准确率计算不准确
            if 'batch_standard_acc' in epoch_data and epoch_data['batch_standard_acc']:
                # 使用计算出的标准准确率作为替代
                train_acc = sum(epoch_data['batch_standard_acc']) / len(epoch_data['batch_standard_acc'])
            train_acc_list.append(train_acc)
        else:
            # 标准准确率计算
            train_acc_list.append(train_corrects.double().item()/train_num if hasattr(train_corrects, 'double') else train_corrects/train_num)
        val_loss_list.append(val_loss/val_num)
        val_acc_list.append(val_corrects.double().item()/val_num)
        
        # 计算AUC指标
        from sklearn.metrics import roc_auc_score
        if len(all_val_probs) > 0 and len(all_val_labels) > 0:
            try:
                auc_score = roc_auc_score(all_val_labels, all_val_probs)
                auc_list.append(auc_score)
            except:
                auc_score = 0.0
                auc_list.append(auc_score)
        else:
            auc_score = 0.0
            auc_list.append(auc_score)
        
        # 记录学习率
        lr_list.append(current_lr)

        print(f"Epoch:{epoch+1} Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f}")
        print(f"Epoch:{epoch+1} Val Loss: {val_loss_list[-1]:.4f} Acc: {val_acc_list[-1]:.4f} AUC: {auc_score:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model weights updated.")
        
        # 更新学习率（余弦退火调度器）
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step()  # 余弦退火不需要传入指标
        current_lr = optimizer.param_groups[0]['lr']
        
        # 手动打印学习率变化信息
        if abs(current_lr - old_lr) > 1e-7:
            print(f"Learning rate updated: {old_lr:.6f} → {current_lr:.6f}")

        # 训练耗费时间
        time_use = time.time() - since
        print(f"Time used: {time_use//60:.0f}m {time_use%60:.0f}s")
        print(f"Best val Acc: {best_acc:.4f}")

    # 保存最优模型参数
    model_save_path = r'best.pth'
    torch.save(best_model_wts, model_save_path)
    print(f"模型已保存到: {model_save_path}")

    # 记录训练过程
    train_process = pd.DataFrame(data={
        'epoch':range(num_epochs),
        'train_loss':train_loss_list,
        'val_loss':val_loss_list,
        'train_acc':train_acc_list,
        'val_acc':val_acc_list,
        'val_auc':auc_list,
        'learning_rate':lr_list})
    
    # 保存详细训练数据
    save_detailed_training_data(detailed_training_data)
    
    # 提取并保存特征数据（使用验证集）
    print("正在提取和保存特征数据...")
    extract_and_save_features(model, val_loader, device)
        
    return train_process

def save_detailed_training_data(detailed_data, save_folder="数据保存"):
    """
    保存详细的训练数据，包括每个batch的损失值
    
    参数:
        detailed_data: 包含详细训练数据的字典列表
        save_folder: 保存文件夹路径
    """
    os.makedirs(save_folder, exist_ok=True)
    
    # 转换为DataFrame
    df = pd.DataFrame(detailed_data)
    
    # 保存详细数据CSV
    csv_path = os.path.join(save_folder, "detailed_training_data_clean.csv")
    df.to_csv(csv_path, index=False)
    print(f"详细训练数据已保存至: {csv_path}")
    
    # 保存损失函数统计信息
    loss_stats = {
        'avg_train_loss': df[df['phase'] == 'train']['loss'].mean(),
        'avg_val_loss': df[df['phase'] == 'val']['loss'].mean(),
        'total_samples': len(df),
        'train_samples': len(df[df['phase'] == 'train']),
        'val_samples': len(df[df['phase'] == 'val']),
        'min_loss': df['loss'].min(),
        'max_loss': df['loss'].max(),
        'loss_std': df['loss'].std()
    }
    
    stats_path = os.path.join(save_folder, "loss_stats_clean.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(loss_stats, f, indent=2, ensure_ascii=False)
    print(f"损失函数统计已保存至: {stats_path}")
    
    return csv_path

def extract_and_save_features(model, data_loader, device, save_folder="数据保存", max_samples=1000):
    """
    提取并保存特征数据（图像特征）
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        save_folder: 保存文件夹
        max_samples: 最大样本数（防止内存不足）
    """
    model.eval()
    
    # 创建临时的特征提取模型
    feature_model = ResNet1(block=Residual, num_classes=2, in_channels=1).to(device)
    feature_model.load_state_dict(model.state_dict())
    
    features_list = []
    labels_list = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_data in data_loader:
            if sample_count >= max_samples:
                break
                
            x, y = batch_data
            x = x.to(device)
            
            # 提取图像特征（到分类层之前）
            features = feature_model.initial(x)
            features = feature_model.layer1(features)
            features = feature_model.layer2(features)
            features = feature_model.layer3(features)
            features = feature_model.layer4(features)
            features = feature_model.layer5[:-2](features)  # 去掉最后的分类层
            features = features.view(features.size(0), -1)  # 展平
            
            # 转换为numpy并保存
            features_list.append(features.cpu().numpy())
            labels_list.append(y.numpy())
            
            sample_count += len(y)
    
    # 合并所有特征
    if features_list:
        features_matrix = np.vstack(features_list)
        labels_array = np.concatenate(labels_list)
        
        # 保存稀疏矩阵数据
        os.makedirs(save_folder, exist_ok=True)
        
        # 图像特征稀疏矩阵（使用CSR格式）
        img_sparse = sparse.csr_matrix(features_matrix)
        sparse.save_npz(os.path.join(save_folder, "image_features_sparse_clean.npz"), img_sparse)
        
        # 保存对应的标签信息
        pd.DataFrame({
            'labels': labels_array
        }).to_csv(os.path.join(save_folder, "feature_metadata_clean.csv"), index=False)
        
        # 保存特征信息统计
        feature_info = {
            'image_features_shape': features_matrix.shape,
            'image_features_sparsity': float(1 - np.count_nonzero(features_matrix) / features_matrix.size),
            'label_distribution': {
                'class_0': int(np.sum(labels_array == 0)),
                'class_1': int(np.sum(labels_array == 1))
            },
            'feature_range': [float(features_matrix.min()), float(features_matrix.max())],
            'feature_mean': float(features_matrix.mean()),
            'feature_std': float(features_matrix.std())
        }
        
        with open(os.path.join(save_folder, "feature_statistics_clean.json"), 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, ensure_ascii=False)
        
        print(f"特征数据已保存至: {save_folder}")
        print(f"  - 图像特征矩阵: {features_matrix.shape}, 稀疏度: {feature_info['image_features_sparsity']:.4f}")
        print(f"  - 特征范围: [{feature_info['feature_range'][0]:.4f}, {feature_info['feature_range'][1]:.4f}]")
        print(f"  - 特征均值: {feature_info['feature_mean']:.4f}, 标准差: {feature_info['feature_std']:.4f}")
    
    return save_folder

def matplot_acc_loss(train_process, save_path="training_curves.pdf"):
    """
    绘制并保存训练过程中的损失和准确率曲线
    
    参数:
        train_process: 包含训练过程的字典，应有'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'键
        save_path: 图片保存路径，默认为"training_curves.pdf"
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], 'bs-', label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc"], 'ro-', label="Train Acc")
    plt.plot(train_process["epoch"], train_process["val_acc"], 'bs-', label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局，确保元素不重叠
    plt.tight_layout()

    # 保存图片为PDF格式
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    print(f"训练曲线已保存至: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    if hasattr(torch, 'npu'):
        print("当前昇腾NPU可用，使用NPU进行训练")
        print("PyTorch版本:", torch.__version__)
        print("NPU设备数量:", torch.npu.device_count() if hasattr(torch.npu, 'device_count') else "未知")
    elif torch.cuda.is_available():
        print("当前 GPU 名称:", torch.cuda.get_device_name())
        print("当前 GPU 索引:", torch.cuda.current_device())
    else:
        print("当前未检测到可用 GPU 或 NPU，使用 CPU 进行训练。")
    
    # 训练参数配置
    num_epochs = 20
    use_mixup = False         # 已移除Mixup数据增强
    mixup_alpha = 0.4         # 保留参数（未使用）
    use_cutmix = False        # 已移除CutMix数据增强
    cutmix_alpha = 1.0        # 保留参数（未使用）
    use_adaptive_aug = False  # 禁用自适应增强
    
    print(f"\n=== 训练配置（简化数据增强） ===")
    print(f"理论相变点: 1.15")
    print(f"训练轮数: {num_epochs}")
    print(f"数据增强策略: 简化几何变换")
    print(f"  - 几何变换: 水平/垂直翻转、旋转、平移、缩放、剪切（减小强度）")
    print(f"  - 移除功能: 高斯噪声、温度扰动、Cutout、Mixup、CutMix")
    print(f"期望效果: 减少过度增强，保持数据原始特征")
    print(f"========================================\n")
    
    train_loader, val_loader = train_val_data_process()
    model = ResNet1(block=Residual, num_classes=2, in_channels=1)
    train_process = train_model_process(model, train_loader, val_loader, num_epochs, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha, use_adaptive_aug)
    matplot_acc_loss(train_process)