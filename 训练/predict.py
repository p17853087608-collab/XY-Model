import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度显示
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(f"{desc}")
        return iterable
from model import ResNet1
from model import Residual

def load_model(model_path=r'best.pth', num_classes=2):
    """加载用户定义的ResNet模型及参数"""
    # 创建用户模型实例
    model = ResNet1(block=Residual, num_classes=num_classes)
    
    # 加载模型参数
    try:
        # 兼容昇腾NPU和标准PyTorch
        if hasattr(torch, 'npu'):
            # 昇腾环境
            model.load_state_dict(torch.load(model_path))
        elif torch.cuda.is_available():
            # GPU环境
            model.load_state_dict(torch.load(model_path))
        else:
            # CPU环境
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"成功加载模型参数: {model_path}")
    except Exception as e:
        print(f"加载模型参数失败: {str(e)}", file=sys.stderr)
        raise
    
    # 设置为评估模式
    model.eval()
    return model

def predict_folder(model, folder_path, output_folder="save_data"):
    """预测单个文件夹中的所有图片"""
    
    # 检查图片文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：图片文件夹不存在: {folder_path}", file=sys.stderr)
        return None
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 定义图片预处理（与训练时保持一致）
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # 从224改为512或更高
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    
    # 获取所有图片文件并按数字顺序排序
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    except Exception as e:
        print(f"读取图片文件夹失败: {str(e)}", file=sys.stderr)
        return None
    
    # 按文件名中的数字排序
    def sort_key(x):
        name_without_ext = os.path.splitext(x)[0]
        try:
            # 提取文件名中的数字部分
            parts = name_without_ext.split('_')
            if len(parts) >= 2:
                return int(parts[-1])  # 取最后一部分作为数字
            return int(name_without_ext)
        except ValueError:
            return name_without_ext
    
    image_files.sort(key=sort_key)
    
    # 检测昇腾NPU环境并设置推理batch size
    if hasattr(torch, 'npu'):
        device = torch.device("npu:0")
        batch_size = 32  # 昇腾NPU推理优化batch size
        print("检测到昇腾NPU，使用NPU进行推理")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        batch_size = 64  # GPU推理batch size
        print("检测到GPU，使用GPU进行推理")
    else:
        device = torch.device("cpu")
        batch_size = 16  # CPU推理batch size
        print("使用CPU进行推理")
    
    model = model.to(device)
    
    folder_name = os.path.basename(folder_path)
    print(f"开始预测文件夹 {folder_name}，共 {len(image_files)} 张图片，使用设备: {device}, batch_size: {batch_size}")
    
    results = []
    
    # 批量预测以提高效率
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"预测 {folder_name}"):
            batch_files = image_files[i:i + batch_size]
            batch_tensors = []
            batch_img_ids = []
            
            # 预处理当前batch
            for img_file in batch_files:
                img_id = os.path.splitext(img_file)[0]
                img_path = os.path.join(folder_path, img_file)
                
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = preprocess(image)
                    batch_tensors.append(image_tensor)
                    batch_img_ids.append(img_id)
                except Exception as e:
                    print(f"预处理图片 {img_file} 时出错: {str(e)}", file=sys.stderr)
                    results.append({
                        "image_id": img_id,
                        "class": "error",
                        "prediction": -1,
                        "ordered_probability": 0.0,
                        "amorphous_probability": 0.0
                    })
            
            if batch_tensors:
                # 批量推理
                batch_tensor = torch.stack(batch_tensors).to(device)
                
                # 批量预测
                outputs = model(batch_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # 根据设备类型处理张量
                if hasattr(torch, 'npu') and next(model.parameters()).is_npu:
                    probabilities = probabilities.cpu()
                elif torch.cuda.is_available() and next(model.parameters()).is_cuda:
                    probabilities = probabilities.cpu()
                probabilities = probabilities.numpy()
                predicted = torch.argmax(outputs, 1).cpu().numpy()
                
                # 处理batch结果
                for j, (img_id, prob, pred) in enumerate(zip(batch_img_ids, probabilities, predicted)):
                    amorphous_prob = round(prob[0], 6)
                    ordered_prob = round(prob[1], 6)
                    class_name = "Amorphous phase" if pred == 0 else "Ordered phase"
                    
                    results.append({
                        "image_id": img_id,
                        "class": class_name,
                        "prediction": int(pred),
                        "ordered_probability": ordered_prob,
                        "amorphous_probability": amorphous_prob
                    })
    
    # 计算统计信息
    if results:
        valid_results = [r for r in results if r["prediction"] != -1]
        
        if valid_results:
            # 计算平均概率和方差
            amorphous_probs = [r["amorphous_probability"] for r in valid_results]
            ordered_probs = [r["ordered_probability"] for r in valid_results]
            
            stats = {
                "folder_name": folder_name,
                "total_images": len(results),
                "valid_predictions": len(valid_results),
                "error_count": len(results) - len(valid_results),
                "avg_ordered_probability": round(np.mean(ordered_probs), 6),
                "avg_amorphous_probability": round(np.mean(amorphous_probs), 6),
                "var_ordered_probability": round(np.var(ordered_probs), 6),
                "var_amorphous_probability": round(np.var(amorphous_probs), 6),
                "amorphous_prediction_rate": round(sum(1 for r in valid_results if r["prediction"] == 1) / len(valid_results), 6)
            }
            
            # 保存详细结果
            output_file = os.path.join(output_folder, f"{folder_name}.csv")
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            
            # 保存统计结果
            stats_file = os.path.join(output_folder, f"{folder_name}_stats.csv")
            stats_df = pd.DataFrame([stats])
            stats_df.to_csv(stats_file, index=False)
            
            print(f"文件夹 {folder_name} 预测完成:")
            print(f"  - 详细结果保存到: {output_file}")
            print(f"  - 统计结果保存到: {stats_file}")
            print(f"  - 平均有序相概率: {stats['avg_ordered_probability']:.6f} (方差: {stats['var_ordered_probability']:.6f})")
            print(f"  - 平均非晶相概率: {stats['avg_amorphous_probability']:.6f} (方差: {stats['var_amorphous_probability']:.6f})")
            print(f"  - 非晶相预测比例: {stats['amorphous_prediction_rate']:.2%}")
            
            return results, stats
        else:
            print(f"文件夹 {folder_name} 中没有有效的预测结果")
            return results, None
    else:
        print(f"文件夹 {folder_name} 中没有找到图片")
        return None, None

def predict_all_folders(base_path=r'32x32_data_set\test', output_folder="save_data"):
    """预测所有文件夹中的图片"""
    
    # 检查基础路径是否存在
    if not os.path.exists(base_path):
        print(f"错误：基础路径不存在: {base_path}", file=sys.stderr)
        return
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有文件夹
    try:
        all_items = os.listdir(base_path)
        folders = [item for item in all_items if os.path.isdir(os.path.join(base_path, item)) and item != output_folder]
        folders.sort()  # 按字母顺序排序
        
        print(f"找到 {len(folders)} 个文件夹需要预测")
        
    except Exception as e:
        print(f"读取文件夹失败: {str(e)}", file=sys.stderr)
        return
    
    # 加载模型
    print("正在加载模型...")
    model = load_model()
    
    # 预测每个文件夹
    all_stats = []
    
    for folder in tqdm(folders, desc="处理文件夹"):
        folder_path = os.path.join(base_path, folder)
        results, stats = predict_folder(model, folder_path, output_folder)
        
        if stats:
            all_stats.append(stats)
    
    # 保存总体统计结果
    if all_stats:
        overall_stats_file = os.path.join(output_folder, "overall_summary.csv")
        overall_df = pd.DataFrame(all_stats)
        overall_df.to_csv(overall_stats_file, index=False)
        print(f"\n所有文件夹预测完成！总体统计保存到: {overall_stats_file}")
        
        # 打印总体统计
        print(f"\n总体统计:")
        print(f"  - 处理文件夹数量: {len(all_stats)}")
        print(f"  - 总图片数量: {sum(s['total_images'] for s in all_stats)}")
        print(f"  - 平均非晶相预测率: {np.mean([s['amorphous_prediction_rate'] for s in all_stats]):.2%}")
    else:
        print("没有成功预测任何文件夹")

if __name__ == "__main__":    
    # 执行预测
    predict_all_folders()