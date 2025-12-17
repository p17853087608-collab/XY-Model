#!/usr/bin/env python3
"""
纯净版自旋配置可视化脚本
生成无任何冗杂信息的纯灰度图
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def process_spin_files_clean():
    """处理当前目录下的所有自旋配置文件，生成纯净图片"""
    
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
    output_dir = current_dir / 'clean_spin_images'
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
                
                # 创建纯净图片 - 无边框、无标题、无颜色条
                plt.figure(figsize=spin_data.shape[::-1])  # 使用数据尺寸作为图片尺寸
                ax = plt.gca()
                ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
                
                # 移除所有冗杂信息
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
                # 保存纯净图片，无边距
                output_file = output_dir / f"{file.stem}.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0)
                plt.close()
                
                print(f"   ✅ 保存纯净图片: {output_file}")
                
            else:
                print(f"   ❌ 不支持的数组维度: {spin_data.shape}")
                
        except Exception as e:
            print(f"   ❌ 处理失败: {e}")
    
    print(f"\n🎉 完成！所有纯净图片保存在: {output_dir}")

if __name__ == "__main__":
    process_spin_files_clean()