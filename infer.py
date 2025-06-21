import torch
import time
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import os
import argparse

# 1. 定义模型
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ImprovedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.se = SEBlock(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.se(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. 加载模型（含torch.compile优化）
def load_model(checkpoint_path, device='cpu'):
    model = ImprovedResNet(num_classes=10)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)
    
    # PyTorch 2.0: 启用模型编译以加速推理（仅支持CUDA/CPU，需显卡驱动兼容）
    if device == 'cuda' and torch.cuda.is_available():
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        print("模型已启用CUDA编译加速")
    elif device == 'cpu':
        model = torch.compile(model, mode="default")
        print("模型已启用CPU编译加速")
    
    return model

# 3. 图像预处理
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    return input_tensor

# 4. 推理与结果解析（使用inference_mode）
def predict_image(model, image_tensor, device='cpu'):
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # PyTorch 2.0推荐：使用inference_mode替代no_grad()
    with torch.inference_mode():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, 1)
    
    result = {
        "class": class_names[top_class.item()],
        "confidence": top_prob.item(),
        "all_probs": probs.numpy()[0].tolist()
    }
    return result

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Classification Inference (PyTorch 2.0)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/best_checkpoint.pth', help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    args = parser.parse_args()
    
    # 1. 加载模型
    model = load_model(args.checkpoint_path, args.device)
    print(f"模型加载自: {args.checkpoint_path}, 设备: {args.device}")
    
    # 2. 预处理图像
    input_tensor = preprocess_image(args.image_path)
    print(f"图像加载: {args.image_path}")
    
    # 3. 推理
    start_time = time.perf_counter()
    result = predict_image(model, input_tensor, args.device)
    inference_time = time.perf_counter() - start_time
    
    # 4. 输出结果
    print(f"预测类别: {result['class']}, 置信度: {result['confidence']:.4f}")
    print(f"推理时间: {inference_time:.4f} 秒")
    print(f"各类别概率: {result['all_probs']}")