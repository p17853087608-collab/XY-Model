#!/usr/bin/env python3
"""
简化版自旋配置可视化脚本
直接在脚本所在目录扫描并生成图片
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_spin_files():
    """处理当前目录下的所有自旋配置文件"""
    
    # 获取当前目录
    current_dir = Path('.')
    
    print("🔍 扫描当前目录中的自旋配置文件...")
    
    # 查找所有.npy文件
    spin_files = []
    for file in current_dir.glob('*.npy'):
        if any(keyword in file.name.lower() for keyword in ['spin', 'config', 'state']):
            spin_files.append(file)
    
    if not spin_files:
        print("❌ 未找到自旋配置文件")
        return
    
    print(f"✅ 找到 {len(spin_files)} 个自旋配置文件")
    
    # 创建输出目录
    output_dir = current_dir / 'spin_images'
    output_dir.mkdir(exist_ok=True)
    
    # 处理每个文件
    for i, file in enumerate(spin_files, 1):
        print(f"🖼️  [{i}/{len(spin_files)}] 处理: {file.name}")
        
        try:
            # 加载数据
            spin_data = np.load(file)
            print(f"   数据形状: {spin_data.shape}")
            
            # 处理2D角度数组
            if spin_data.ndim == 2:
                # 归一化到[0, 2π]
                angles = spin_data % (2 * np.pi)
                grayscale = angles / (2 * np.pi)
                
                # 生成纯净图片 - 无边框、无标题、无颜色条
                plt.figure(figsize=(6, 6))
                ax = plt.gca()
                ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
                
                # 提取温度信息
                temp_str = "unknown"
                if "T_" in file.name:
                    temp_part = file.name.split("T_")[1].split("_")[0]
                    temp_str = f"T={temp_str}"
                
                plt.title(f'Spin Configuration ({temp_str})')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.tight_layout()
                
                # 保存图片
                output_file = output_dir / f"{file.stem}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ 保存图片: {output_file}")
                
            else:
                print(f"   ❌ 不支持的数组维度: {spin_data.shape}")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print(f"\n🎉 完成！所有图片保存在: {output_dir}")

if __name__ == "__main__":
    process_spin_files()