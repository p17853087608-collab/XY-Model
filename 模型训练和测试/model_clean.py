import torch 
import torch.nn as nn
try:
    from torchsummary import summary
except ImportError:
    print("警告：torchsummary未安装，无法显示模型摘要")
    def summary(model, input_size):
        print(f"模型：{model.__class__.__name__}")
        print(f"输入尺寸：{input_size}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数数量：{total_params:,}")
        print(f"可训练参数：{trainable_params:,}")
        return None

class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        super(Residual,self).__init__()
        self.ReLU = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels,kernel_size=3,stride=strides,padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)    
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
    
    def forward(self,x):
        Y = self.ReLU(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        Y += x
        Y = self.ReLU(Y)
        return Y

class ResNet1(nn.Module):
    """标准ResNet模型"""
    def __init__(self,block,num_classes=10,in_channels=1):
        super(ResNet1, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.Sequential(
            block(input_channels=64, num_channels=64, use_1x1conv=False, strides=1),
            block(input_channels=64, num_channels=64, use_1x1conv=False, strides=1),
        )
        self.layer2 = nn.Sequential(
            block(input_channels=64, num_channels=128, use_1x1conv=True, strides=2),
            block(input_channels=128, num_channels=128, use_1x1conv=False, strides=1),
        )
        self.layer3 = nn.Sequential(
            block(input_channels=128, num_channels=256, use_1x1conv=True, strides=2),
            block(input_channels=256, num_channels=256, use_1x1conv=False, strides=1),
        )
        self.layer4 = nn.Sequential(
            block(input_channels=256, num_channels=512, use_1x1conv=True, strides=2),
            block(input_channels=512, num_channels=512, use_1x1conv=False, strides=1),
        )
        self.layer5 = nn.Sequential(        
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

if __name__ == "__main__":
    # 检测昇腾NPU环境
    if hasattr(torch, 'npu'):
        device = torch.device("npu:0")
        print("使用昇腾NPU")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用GPU")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    model = ResNet1(block=Residual, num_classes=2)
    model.to(device)
    print(summary(model, (1, 224, 224)))