#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
二维XY模型磁化率统计分析脚本
统计所有npy文件并绘制磁化率随温度变化的曲线
支持自定义格点尺寸
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from scipy.ndimage import convolve
import matplotlib
import platform
from collections import defaultdict
import argparse

# 设置中文字体，根据不同系统自动选择
system = platform.system()
if system == 'Windows':
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
elif system == 'Darwin':  # macOS
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS', 'DejaVu Sans', 'Arial']
else:  # Linux
    matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK', 'DejaVu Sans', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'


def bootstrap_susceptibility_error(M_norm_list, M_sq_list, temp, N, n_bootstrap=1000, random_seed=None):
    """
    使用bootstrap方法估计磁化率的标准差
    
    Args:
        M_norm_list: 该温度下所有样本的|M|列表
        M_sq_list: 该温度下所有样本的|M|²列表
        temp: 温度
        N: 总格点数
        n_bootstrap: bootstrap重采样次数
        random_seed: 随机种子（用于可重复性）
    
    Returns:
        chi: 磁化率
        chi_std: 磁化率的标准差
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(M_norm_list)
    chi_bootstrap = []
    
    M_norms = np.array(M_norm_list)
    M_sqs = np.array(M_sq_list)
    
    for _ in range(n_bootstrap):
        # 重采样
        indices = np.random.choice(n_samples, n_samples, replace=True)
        
        # 计算重采样后的磁化率
        M_norm_resampled = M_norms[indices]
        M_sq_resampled = M_sqs[indices]
        
        avg_M_norm_boot = np.mean(M_norm_resampled)
        avg_M_sq_boot = np.mean(M_sq_resampled)
        
        chi_boot = (avg_M_sq_boot - avg_M_norm_boot**2) / (temp * N)
        chi_bootstrap.append(chi_boot)
    
    # 计算原始磁化率
    avg_M_norm = np.mean(M_norms)
    avg_M_sq = np.mean(M_sqs)
    chi = (avg_M_sq - avg_M_norm**2) / (temp * N)
    
    # 计算bootstrap标准差
    chi_std = np.std(chi_bootstrap)
    
    return chi, chi_std


def extract_temperature(filename):
    """从文件名中提取温度值"""
    match = re.search(r'T_([\d.]+)', filename)
    if match:
        return float(match.group(1))
    return None


def calculate_magnetization(spins):
    """
    计算单个自旋构型的总磁化强度
    返回: M_x, M_y (磁化强度的两个分量), M_sq (|M|²)
    """
    # 计算总磁化强度矢量的两个分量
    M_x = np.sum(np.cos(spins))
    M_y = np.sum(np.sin(spins))
    
    # 计算总磁化强度的平方 (|M|²)
    M_sq = M_x**2 + M_y**2
    
    return M_x, M_y, M_sq


def process_npy_files(base_path, lattice_size):
    """
    处理指定路径下的所有npy文件，计算磁化率
    对每个温度点的多个样本计算平均磁化率
    
    Args:
        base_path: 数据文件路径
        lattice_size: 格点尺寸 (如 256 或 128)
    """
    base_path = Path(base_path)
    
    # 使用字典按温度分组存储数据
    temp_data = defaultdict(list)
    
    # 递归搜索所有npy文件
    for npy_file in base_path.rglob("*.npy"):
        temp = extract_temperature(npy_file.name)
        if temp is not None:
            temp_data[temp].append(npy_file)
    
    # 按温度排序
    sorted_temps = sorted(temp_data.keys())
    
    print(f"找到 {len(sorted_temps)} 个不同温度点")
    print(f"每个温度点的样本数: {[len(temp_data[t]) for t in sorted_temps[:10]]}...")  # 只显示前10个
    
    # 计算每个温度点的平均磁化率
    temperatures = []
    avg_magnetizations = []
    avg_susceptibilities = []
    std_magnetizations = []
    std_susceptibilities = []
    
    total_susceptibilities = []  # 用于计算整体平均磁化率
    
    N = lattice_size * lattice_size  # 总格点数
    
    for temp in sorted_temps:
        files = temp_data[temp]
        
        # 收集该温度下所有样本的|M|和|M|²
        M_norm_list = []  # 存储每个构型的|M|
        M_sq_list = []   # 存储每个构型的|M|²
        
        for npy_file in files:
            try:
                spins = np.load(npy_file)
                M_x, M_y, M_sq = calculate_magnetization(spins)
                M_norm = np.sqrt(M_sq)
                
                M_norm_list.append(M_norm)
                M_sq_list.append(M_sq)
                
            except Exception as e:
                print(f"处理 {npy_file.name} 时出错: {e}")
                continue
        
        if len(M_norm_list) > 0:
            # 使用bootstrap方法计算磁化率及其标准差
            avg_susceptibility, std_susceptibility = bootstrap_susceptibility_error(
                M_norm_list, M_sq_list, temp, N, n_bootstrap=1000, random_seed=42
            )
            
            # 计算磁化强度的平均值和标准差
            avg_M_norm = np.mean(M_norm_list)  # <|M|⟩
            std_M_norm = np.std(M_norm_list)
            
            temperatures.append(temp)
            avg_magnetizations.append(avg_M_norm)
            avg_susceptibilities.append(avg_susceptibility)
            std_magnetizations.append(std_M_norm)
            std_susceptibilities.append(std_susceptibility)
            
            print(f"T={temp:.3f}: {len(M_norm_list)}个样本, 平均|M|={avg_M_norm:.4f}±{std_M_norm:.4f}, 平均χ={avg_susceptibility*1000:.4f}±{std_susceptibility*1000:.4f}x10^-3")
    
    # 转换为numpy数组
    temperatures = np.array(temperatures)
    avg_magnetizations = np.array(avg_magnetizations)
    avg_susceptibilities = np.array(avg_susceptibilities)
    std_magnetizations = np.array(std_magnetizations)
    std_susceptibilities = np.array(std_susceptibilities)
    
    # 计算整个温度区间的整体平均磁化率
    overall_avg_susceptibility = np.mean(avg_susceptibilities)
    overall_std_susceptibility = np.mean(std_susceptibilities)
    
    return temperatures, avg_magnetizations, avg_susceptibilities, std_magnetizations, std_susceptibilities, overall_avg_susceptibility, overall_std_susceptibility


def plot_avg_susceptibility_vs_temp(temperatures, avg_susceptibilities, std_susceptibilities, overall_avg, save_path, lattice_size):
    """
    绘制平均磁化率随温度变化的曲线
    x轴：温度
    y轴：每个温度点的平均磁化率
    
    Args:
        lattice_size: 格点尺寸，用于标题显示
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制曲线和误差棒
    ax.errorbar(temperatures, avg_susceptibilities * 1000, yerr=std_susceptibilities * 1000,
                fmt='o-', linewidth=2.5, markersize=5, alpha=0.7, color='red',
                capsize=3, capthick=1.5, ecolor='darkred', label=r'Average Susceptibility $\chi$')
    
    # 添加填充效果（标准差范围）
    ax.fill_between(temperatures, 
                    (avg_susceptibilities - std_susceptibilities) * 1000,
                    (avg_susceptibilities + std_susceptibilities) * 1000,
                    alpha=0.15, color='red', label='Standard Deviation')
    
    # 绘制整体平均磁化率的水平线
    ax.axhline(y=overall_avg * 1000, color='blue', linestyle='--', linewidth=2, alpha=0.8,
               label=r'Overall Average: $\chi$={:.4f}$\times$10$^{{-3}}$'.format(overall_avg*1000))
    
    # 标记峰值点
    max_idx = np.argmax(avg_susceptibilities)
    ax.plot(temperatures[max_idx], avg_susceptibilities[max_idx] * 1000, 'g*', 
            markersize=20, label=r'Peak: T={:.3f}, $\chi$={:.4f}$\times$10$^{{-3}}$'.format(
                temperatures[max_idx], avg_susceptibilities[max_idx]*1000),
            zorder=5)
    
    # 标记峰值垂直线
    ax.axvline(x=temperatures[max_idx], color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Temperature T', fontsize=16)
    ax.set_ylabel(r'Average Susceptibility $\chi$ ($\times$10$^{-3}$)', fontsize=16)
    ax.set_title(f'2D XY Model ({lattice_size}x{lattice_size}): Average Susceptibility vs Temperature', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=13)
    
    # 设置坐标轴范围
    ax.set_xlim(temperatures.min() - 0.01, temperatures.max() + 0.01)
    
    # 添加文本框显示统计信息
    textstr = r'Overall Average $\chi$: {:.4f}$\times$10$^{{-3}}$\nStd Dev: {:.4f}$\times$10$^{{-3}}$\nPeak at T={:.3f}'.format(
        overall_avg*1000, np.mean(std_susceptibilities)*1000, temperatures[max_idx])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAverage susceptibility plot saved to: {save_path}")
    plt.close()


def plot_magnetization_with_error(temperatures, avg_magnetizations, std_magnetizations, save_path, lattice_size):
    """
    绘制平均磁化强度随温度变化的曲线（带误差棒）
    
    Args:
        lattice_size: 格点尺寸，用于标题显示
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 绘制曲线和误差棒
    ax.errorbar(temperatures, avg_magnetizations, yerr=std_magnetizations,
                fmt='o-', linewidth=2.5, markersize=5, alpha=0.7, color='blue',
                capsize=3, capthick=1.5, ecolor='darkblue', label='Average Magnetization |M|')
    
    # 添加填充效果（标准差范围）
    ax.fill_between(temperatures, 
                    avg_magnetizations - std_magnetizations,
                    avg_magnetizations + std_magnetizations,
                    alpha=0.15, color='blue', label='Standard Deviation')
    
    # 标记最大值点
    max_idx = np.argmax(avg_magnetizations)
    ax.plot(temperatures[max_idx], avg_magnetizations[max_idx], 'r*', 
            markersize=20, label=f'Max: T={temperatures[max_idx]:.3f}, |M|={avg_magnetizations[max_idx]:.4f}',
            zorder=5)
    
    ax.set_xlabel('Temperature T', fontsize=16)
    ax.set_ylabel('Average Magnetization |M|', fontsize=16)
    ax.set_title(f'2D XY Model ({lattice_size}x{lattice_size}): Average Magnetization vs Temperature', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.legend(fontsize=12, loc='best')
    ax.tick_params(axis='both', labelsize=13)
    
    ax.set_xlim(temperatures.min() - 0.01, temperatures.max() + 0.01)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Magnetization plot saved to: {save_path}")
    plt.close()


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Analyze 2D XY model susceptibility from npy files')
    parser.add_argument('--path', type=str, default=".",
                       help='Path to the data directory containing npy files')
    parser.add_argument('--size', type=int, default=256,
                       help='Lattice size (default: 256)', choices=[64, 128, 256, 512])
    
    args = parser.parse_args()
    
    base_path = args.path
    lattice_size = args.size
    
    print("=" * 60)
    print(f"2D XY Model Susceptibility Analysis ({lattice_size}x{lattice_size})")
    print("=" * 60)
    print(f"Data path: {base_path}")
    print(f"Lattice size: {lattice_size}x{lattice_size}")
    
    # Process all npy files
    (temperatures, avg_magnetizations, avg_susceptibilities, 
     std_magnetizations, std_susceptibilities, 
     overall_avg_susceptibility, overall_std_susceptibility) = process_npy_files(base_path, lattice_size)
    
    if len(temperatures) == 0:
        print("No valid npy files found!")
        return
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)
    print(f"Temperature range: {temperatures.min():.3f} - {temperatures.max():.3f}")
    print(f"Average magnetization range: {avg_magnetizations.min():.4f} - {avg_magnetizations.max():.4f}")
    print(f"Average magnetization (all): {avg_magnetizations.mean():.4f}")
    print(f"Average susceptibility range (x10^-3): {avg_susceptibilities.min()*1000:.4f} - {avg_susceptibilities.max()*1000:.4f}")
    print(f"\nOVERALL AVERAGE SUSCEPTIBILITY: {overall_avg_susceptibility*1000:.6f} x10^-3")
    print(f"Overall susceptibility std dev: {overall_std_susceptibility*1000:.6f} x10^-3")
    
    # Find peak point
    max_sus_idx = np.argmax(avg_susceptibilities)
    print(f"\nPeak point:")
    print(f"  Temperature: {temperatures[max_sus_idx]:.3f}")
    print(f"  Susceptibility: {avg_susceptibilities[max_sus_idx]*1000:.4f} x10^-3")
    print(f"  Std Dev: {std_susceptibilities[max_sus_idx]*1000:.4f} x10^-3")
    
    # Save results to text file
    results_file = Path(base_path) / "susceptibility_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"# 2D XY Model ({lattice_size}x{lattice_size}) Susceptibility Analysis\n")
        f.write("# Temperature\tAvg_Magnetization\tMagnetization_Std\tAvg_Susceptibility(x10^-3)\tSusceptibility_Std(x10^-3)\n")
        for i in range(len(temperatures)):
            f.write(f"{temperatures[i]:.6f}\t{avg_magnetizations[i]:.6f}\t{std_magnetizations[i]:.6f}\t"
                    f"{avg_susceptibilities[i]*1000:.6f}\t{std_susceptibilities[i]*1000:.6f}\n")
        f.write(f"\n# Overall Average Susceptibility: {overall_avg_susceptibility*1000:.6f} x10^-3\n")
        f.write(f"# Overall Susceptibility Std Dev: {overall_std_susceptibility*1000:.6f} x10^-3\n")
    print(f"\nResults saved to: {results_file}")
    
    # Save raw data to npy file
    np.savez(Path(base_path) / "susceptibility_data.npz",
             lattice_size=lattice_size,
             temperatures=temperatures,
             avg_magnetizations=avg_magnetizations,
             std_magnetizations=std_magnetizations,
             avg_susceptibilities=avg_susceptibilities,
             std_susceptibilities=std_susceptibilities,
             overall_avg_susceptibility=overall_avg_susceptibility,
             overall_std_susceptibility=overall_std_susceptibility)
    print(f"Raw data saved to: {Path(base_path) / 'susceptibility_data.npz'}")
    
    # Generate plots
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    plot_magnetization_with_error(temperatures, avg_magnetizations, std_magnetizations,
                                   Path(base_path) / "avg_magnetization_plot.png", lattice_size)
    
    plot_avg_susceptibility_vs_temp(temperatures, avg_susceptibilities, std_susceptibilities,
                                    overall_avg_susceptibility,
                                    Path(base_path) / "avg_susceptibility_vs_temp.png", lattice_size)
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
