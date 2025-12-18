import os
import copy
import time
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as Data
import matplotlib.pyplot as plt
from model import ResNet1
from model import Residual
import torch
import pandas as pd

def get_data_path(relative_path):
    """获取跨平台的正确数据路径"""
    return os.path.join(os.getcwd(), relative_path.replace('\\', os.sep))

# 检查PyTorch版本兼容性
if torch.__version__ < '1.9.0':
    print(f"警告：检测到PyTorch版本 {torch.__version__}，较老版本可能存在兼容性问题")

# 数据处理函数，划分训练集和验证集
def train_val_data_process():
    ROOT_PATH = get_data_path('32x32_data_set\\train')
    train_data = transforms.Compose([
        transforms.Resize((224,224)),  # 从224改为512或更高
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道标准化
    ])
    train_data = ImageFolder(root=ROOT_PATH, transform=train_data)
    # 划分训练集和验证集，比例为8:2
    train_data, val_data = Data.random_split(train_data,
                                            [round(len(train_data)*0.8), 
                                            len(train_data)-round(len(train_data)*0.8)])
    # 训练集和验证集的数据加载器 
    # 根据硬件环境优化设置
    if hasattr(torch, 'npu'):
        # 昇腾NPU环境 - 16GB显存可以使用更大batch
        print("昇腾NPU环境，检测到大显存，使用优化batch size")
        train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=0)
    elif torch.cuda.is_available():
        print("GPU环境")
        train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=4)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=4)
    else:
        # CPU环境设置
        print("CPU环境")
        train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True, num_workers=2)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=32, shuffle=True, num_workers=2)

    return train_loader, val_loader

# 训练模型的函数
def train_model_process(model, train_loader, val_loader, num_epochs):
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
    # 损失函数为交叉熵损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 优化器使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 将模型发送到所用设备
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())
    # 深拷贝的作用：copy.deepcopy()会递归地创建对象及其所有子对象的全新副本。
    # 这样一来，best_model_wts就成为一个完全独立的字典，其中的参数张量与模型当前的参数张量彻底脱离关系。
    # 无论模型后续如何训练更新，best_model_wts都雷打不动地保存着执行这行代码时的模型状态

    # 最高准确度
    best_acc = 0.0

    # 训练集损失值列表和验证集损失值列表
    train_loss_list = []
    val_loss_list = []
    # 训练集精度值列表和验证集精度值列表
    train_acc_list = []
    val_acc_list = []

    # 记录训练开始时间
    since = time.time()

    # 开始训练
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-'*10)

        # 初始化参数
        # 损失值 精度值 样本数量
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        train_num = 0
        val_num = 0

        model.train()
        for step,(x,y) in enumerate(train_loader):
            # 特征
            x = x.to(device)
            # 标签
            y = y.to(device)
            
            # 前向传播
            outputs = model(x)
            pre_lab = torch.argmax(outputs, dim=1)

            loss = criterion(outputs, y)

            # 梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算训练集的损失值和精度值
            train_loss += loss.item() * x.size(0)
            train_corrects += torch.sum(pre_lab == y.data)
            train_num += x.size(0)

        for step,(x,y) in enumerate(val_loader):
            
            model.eval()
            # 特征
            x = x.to(device)
            # 标签
            y = y.to(device)
            
            with torch.no_grad():
                # 前向传播
                outputs = model(x)
                pre_lab = torch.argmax(outputs, dim=1)

                loss = criterion(outputs, y)

                # 计算验证集的损失值和精度值
                val_loss += loss.item() * x.size(0)
                val_corrects += torch.sum(pre_lab == y.data)
                val_num += x.size(0)

        # 计算训练集和验证集的loss值
            #获取损失值或准确率：在模型训练过程中，
            # 损失函数（如 criterion(outputs, labels)）返回的结果通常是一个单元素张量。
            # 为了打印这个值、记录到日志中或者进行简单的Python数值比较，
            # 你需要使用 loss.item()将其转换为一个普通的Python数字
        train_loss_list.append(train_loss/train_num)
        train_acc_list.append(train_corrects.double().item()/train_num)
        val_loss_list.append(val_loss/val_num)
        val_acc_list.append(val_corrects.double().item()/val_num)

        print(f"Epoch:{epoch+1} Train Loss: {train_loss_list[-1]:.4f} Acc: {train_acc_list[-1]:.4f}")
        print(f"Epoch:{epoch+1} Val Loss: {val_loss_list[-1]:.4f} Acc: {val_acc_list[-1]:.4f}")

        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            print("Best model weights updated.")

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
        'val_acc':val_acc_list})
        
    return train_process

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
    num_epochs = 20
    train_loader, val_loader = train_val_data_process()
    model = ResNet1(block=Residual, num_classes=2, input_channels=1)
    train_process = train_model_process(model, train_loader, val_loader, num_epochs)
    matplot_acc_loss(train_process)

