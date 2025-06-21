# app.py
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import time

# 导入统一的模型定义
from model import ImprovedResNet50

app = Flask(__name__)

# 全局变量：模型和预处理函数
model = None
transform = None
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_model():
    global model
    try:
        # 使用与训练一致的模型结构
        model = ImprovedResNet50(num_classes=10, use_se=True)
        
        # 模型权重文件路径
        checkpoint_path = 'checkpoints/best_checkpoint.pth'
        
        # 检查模型文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在：{checkpoint_path}")
        
        # 加载权重
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 确保checkpoint包含model_state_dict
        if 'model_state_dict' not in checkpoint:
            raise KeyError("检查点文件中缺少'model_state_dict'键，请确认训练时正确保存模型")
        
        # 加载模型状态 (设置strict=False可以忽略不匹配的键，但不建议长期使用)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model.eval()
        
        # 设备迁移
        if torch.cuda.is_available():
            model = model.cuda()
        
        print("✅ 模型加载成功！")
        
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        raise

def preprocess_image(image_file):
    global transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        ])
    image = Image.open(image_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    return input_tensor

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "未提供图像文件"}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "未选择图像"}), 400
    
    # 预处理图像
    try:
        input_tensor = preprocess_image(image_file)
    except Exception as e:
        return jsonify({"error": f"图像处理失败：{str(e)}"}), 500
    
    # 推理
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        top_prob, top_class = torch.max(probs, 1)
    inference_time = time.time() - start_time
    
    # 构建结果
    result = {
        "class": class_names[top_class.item()],
        "confidence": top_prob.item(),
        "inference_time": inference_time,
        "all_probs": {class_names[i]: float(probs[0, i]) for i in range(10)}
    }
    
    return jsonify(result)

if __name__ == "__main__":
    try:
        print("正在启动图像分类服务...")
        load_model()
        print("服务启动成功，等待请求...")
        app.run(host='127.0.0.1', port=8000, debug=True)
    except Exception as e:
        print(f"服务启动失败：{e}")