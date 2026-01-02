#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG图片分类脚本 (版本2)
根据温度区间起始点判断相类型，并重命名文件
温度区间起始点大于1.0: Amorphous phase
温度区间起始点小于等于1.0: Ordered phase
"""

import os
import shutil
import re
from pathlib import Path

def extract_temperature_from_filename(filename):
    """
    从文件名中提取温度值
    文件名格式: unknown_3.900_spin_config_t_3.900.png 或 spin_config_t_3.900.png
    """
    # 使用正则表达式匹配温度值，支持多种格式
    match = re.search(r't_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"无法从文件名中提取温度值: {filename}")

def extract_random_number_from_path(filepath):
    """
    从文件路径中提取随机数（文件夹名中的数字部分）
    路径格式: .../parallel_simulation_results_YYYYMMDD_HHMMSS_XXXX/...
    """
    # 分割路径，找到包含parallel_simulation_results的部分
    path_parts = filepath.split(os.sep)
    
    for part in path_parts:
        if part.startswith('parallel_simulation_results_'):
            # 提取时间戳后的随机数部分
            # 格式: parallel_simulation_results_YYYYMMDD_HHMMSS_XXXX
            parts = part.split('_')
            if len(parts) >= 4:
                return parts[-1]  # 最后一个部分是随机数
    
    return "unknown"

def determine_phase_by_temperature_start(temperature):
    """
    根据温度判断相类型
    对于2D Ising模型，临界温度约为2.269
    温度大于2.269: Amorphous phase (无序相)
    温度小于等于2.269: Ordered phase (有序相)
    """
    critical_temperature = 2.269
    if temperature > critical_temperature:
        return "Amorphous phase"
    else:
        return "Ordered phase"

def create_new_filename(random_num, temperature, original_filename):
    """
    创建新的文件名，格式: 随机数_温度_原文件名
    """
    # 移除原文件名中的spin_vis_前缀和_raw.npy.png后缀，只保留核心部分
    base_name = original_filename.replace('spin_vis_', '').replace('_raw.npy.png', '')
    
    # 新文件名格式: 随机数_温度_核心名称.png
    new_filename = f"{random_num}_{temperature:.3f}_{base_name}.png"
    
    return new_filename

def process_single_png(png_path, output_base_dir):
    """
    处理单个PNG文件，确定相类型并复制到相应文件夹
    """
    filename = os.path.basename(png_path)
    
    try:
        # 提取温度
        temperature = extract_temperature_from_filename(filename)
        
        # 提取随机数
        random_num = extract_random_number_from_path(png_path)
        
        # 根据温度起始点判断相类型
        phase = determine_phase_by_temperature_start(temperature)
        
        # 创建目标文件夹路径
        phase_folder = os.path.join(output_base_dir, phase)
        
        # 确保目标文件夹存在
        os.makedirs(phase_folder, exist_ok=True)
        
        # 创建新的文件名
        new_filename = create_new_filename(random_num, temperature, filename)
        
        # 目标文件路径
        dest_path = os.path.join(phase_folder, new_filename)
        
        # 复制文件
        shutil.copy2(png_path, dest_path)
        
        print(f"已复制: {filename} -> {phase}/{new_filename} (温度: {temperature})")
        
        return True, phase
        
    except Exception as e:
        print(f"处理文件 {png_path} 时出错: {e}")
        return False, None

def main():
    # 当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 获取所有PNG文件
    print("正在扫描PNG文件...")
    png_files = []
    
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    if not png_files:
        print("未找到任何PNG文件!")
        return
    
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 创建输出目录
    output_base_dir = os.path.join(current_dir, "classified_images")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 统计信息
    phase_stats = {"Amorphous phase": 0, "Ordered phase": 0}
    success_count = 0
    fail_count = 0
    
    # 处理每个文件
    print("\n开始处理PNG文件...")
    for i, png_path in enumerate(png_files, 1):
        print(f"\n处理进度: {i}/{len(png_files)}")
        
        success, phase = process_single_png(png_path, output_base_dir)
        
        if success:
            success_count += 1
            phase_stats[phase] += 1
        else:
            fail_count += 1
    
    # 输出统计结果
    print(f"\n{'='*50}")
    print(f"处理完成!")
    print(f"成功处理: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"\n相分布统计:")
    print(f"  Amorphous phase: {phase_stats['Amorphous phase']} 个文件")
    print(f"  Ordered phase: {phase_stats['Ordered phase']} 个文件")
    print(f"\n目标文件夹: {output_base_dir}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()