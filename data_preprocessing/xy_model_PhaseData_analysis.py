# 整理临界区数据 并且可选择分析结果

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import re
import shutil

# 设置美观的样式（不使用中文字体）
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
rcParams['figure.dpi'] = 120
rcParams['savefig.dpi'] = 300
rcParams['figure.figsize'] = [14, 10]
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 11
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10

def extract_temperature_from_filename(filename):
    """从文件名中提取温度值"""
    match = re.search(r't_(\d+\.\d+)', filename)
    if match:
        return float(match.group(1))
    return None

def create_test_folders():
    """创建test文件夹和对应的温度子文件夹"""
    print("正在扫描温度值...")
    
    # 获取所有PNG文件（支持新旧两种目录结构）
    png_files = []
    
    # 新的parallel_simulation_results结构
    png_files.extend(glob.glob("./parallel_simulation_results_*/spin_configurations/generated_visualizations/*.png"))
    
    # 旧的simulation_results结构（如果存在）
    png_files.extend(glob.glob("./simulation_results_*/spin_configurations/generated_visualizations/*.png"))
    
    print(f"总共找到 {len(png_files)} 个PNG文件")
    
    # 提取所有唯一的温度值
    temperatures = set()
    for png_file in png_files:
        filename = os.path.basename(png_file)
        # 匹配两种格式：spin_vis_spin_config_t_0.750_raw.npy.png 和 t_0.750_*.png
        match = re.search(r't_(\d+\.\d+)', filename)
        if match:
            temperatures.add(float(match.group(1)))
    
    # 排序温度值
    sorted_temps = sorted(temperatures)
    
    if not sorted_temps:
        print("警告: 没有找到任何温度数据！请检查PNG文件路径。")
        return []
    
    print(f"发现 {len(sorted_temps)} 个不同温度值")
    print(f"温度范围: {min(sorted_temps):.3f} - {max(sorted_temps):.3f}")
    
    # 创建test基础目录
    test_base_dir = "./test"
    if not os.path.exists(test_base_dir):
        os.makedirs(test_base_dir)
        print(f"创建了基础目录: {test_base_dir}")
    else:
        print(f"基础目录已存在: {test_base_dir}")
    
    # 创建每个温度对应的子文件夹
    created_folders = []
    for temp in sorted_temps:
        folder_name = f"{temp:.3f}"
        folder_path = os.path.join(test_base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            created_folders.append(folder_name)
    
    print(f"创建了 {len(created_folders)} 个温度子文件夹")
    return sorted_temps

def copy_all_images():
    """复制所有图片到对应的文件夹并重命名"""
    print("正在复制图片文件...")
    
    # 获取所有PNG文件（支持新旧两种目录结构）
    png_files = []
    
    # 新的parallel_simulation_results结构
    png_files.extend(glob.glob("./parallel_simulation_results_*/spin_configurations/generated_visualizations/*.png"))
    
    # 旧的simulation_results结构（如果存在）
    png_files.extend(glob.glob("./simulation_results_*/spin_configurations/generated_visualizations/*.png"))
    
    print(f"找到 {len(png_files)} 个PNG文件准备复制")
    
    # 获取test文件夹中的所有文件夹
    test_base_dir = "./test"
    test_folders = glob.glob(os.path.join(test_base_dir, "*"))
    test_folders = [f for f in test_folders if os.path.isdir(f)]
    test_folders.sort()
    
    # 创建温度到文件夹的映射
    temp_to_folder = {}
    for folder_path in test_folders:
        folder_name = os.path.basename(folder_path)
        try:
            temp_to_folder[float(folder_name)] = folder_path
        except ValueError:
            continue
    
    # 按温度分组所有图片
    from collections import defaultdict
    temp_images = defaultdict(list)
    
    for png_file in png_files:
        filename = os.path.basename(png_file)
        temperature = extract_temperature_from_filename(filename)
        
        if temperature is not None:
            temp_images[temperature].append(png_file)
    
    # 处理每个温度的图片
    total_copied = 0
    for temperature, images in temp_images.items():
        if temperature in temp_to_folder:
            target_folder = temp_to_folder[temperature]
            folder_index = test_folders.index(target_folder) + 1
            
            for i, png_file in enumerate(images, 1):
                new_filename = f"{folder_index}_{i:03d}.png"
                target_path = os.path.join(target_folder, new_filename)
                
                shutil.copy2(png_file, target_path)
                total_copied += 1
        else:
            print(f"  警告: 温度 {temperature:.3f} 没有对应的文件夹")
    
    print(f"成功复制 {total_copied} 张图片")
    return total_copied

def read_simulation_data():
    """读取所有simulation_results文件夹中的数据（支持新旧两种格式）"""
    print("正在读取仿真数据...")
    
    # 获取所有可能的结果目录
    all_sim_dirs = []
    
    # 新的parallel_simulation_results结构
    parallel_dirs = glob.glob("./parallel_simulation_results_*")
    all_sim_dirs.extend(parallel_dirs)
    
    # 旧的simulation_results结构（如果存在）
    old_dirs = glob.glob("./simulation_results_*")
    all_sim_dirs.extend(old_dirs)
    
    print(f"找到 {len(all_sim_dirs)} 个仿真结果目录")
    if parallel_dirs:
        print(f"  - 新格式目录: {len(parallel_dirs)} 个")
    if old_dirs:
        print(f"  - 旧格式目录: {len(old_dirs)} 个")
    
    all_data = {}
    successful_dirs = 0
    
    for sim_dir in all_sim_dirs:
        # 尝试新的数据文件格式
        result_file_new = os.path.join(sim_dir, "results_summary.txt")
        # 尝试旧的数据文件格式
        result_file_old = os.path.join(sim_dir, "simulation_results_optimized.txt")
        
        result_file = None
        data_format = None
        
        if os.path.exists(result_file_new):
            result_file = result_file_new
            data_format = "new"
        elif os.path.exists(result_file_old):
            result_file = result_file_old
            data_format = "old"
        
        if result_file and data_format:
            try:
                data = []
                
                if data_format == "new":
                    # 新格式：手动读取，跳过注释行
                    with open(result_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#'):
                                parts = line.split()
                                if len(parts) >= 5:
                                    try:
                                        temp = float(parts[0])
                                        energy = float(parts[1])
                                        magnetization = float(parts[2])
                                        susceptibility = float(parts[3])
                                        specific_heat = float(parts[4])
                                        data.append([temp, energy, magnetization, susceptibility, specific_heat])
                                    except ValueError:
                                        continue
                else:
                    # 旧格式：直接用numpy加载
                    data = np.loadtxt(result_file)
                    if data.ndim == 2 and data.shape[1] >= 5:
                        data = data.tolist()
                    else:
                        data = []
                
                if data:
                    data = np.array(data)
                    temperatures = data[:, 0]
                    energies = data[:, 1] 
                    
                    if data_format == "new":
                        # 新格式列顺序：温度, 能量, 磁化强度, 磁化率, 比热
                        specific_heats = data[:, 4]  # 第5列是比热
                        magnetizations = data[:, 2]  # 第3列是磁化强度
                        susceptibilities = data[:, 3]  # 第4列是磁化率
                    else:
                        # 旧格式列顺序：温度, 能量, 比热, 磁化强度, 磁化率
                        specific_heats = data[:, 2]  # 第3列是比热
                        magnetizations = data[:, 3]  # 第4列是磁化强度
                        susceptibilities = data[:, 4]  # 第5列是磁化率
                    
                    for i, temp in enumerate(temperatures):
                        if temp not in all_data:
                            all_data[temp] = {
                                'energies': [],
                                'magnetizations': [],
                                'specific_heats': [],
                                'susceptibilities': []
                            }
                        
                        all_data[temp]['energies'].append(energies[i])
                        all_data[temp]['magnetizations'].append(magnetizations[i])
                        all_data[temp]['specific_heats'].append(specific_heats[i])
                        all_data[temp]['susceptibilities'].append(susceptibilities[i])
                    
                    successful_dirs += 1
                    format_name = "新格式" if data_format == "new" else "旧格式"
                    print(f"  成功读取: {sim_dir} ({format_name}, {len(data)} 个数据点)")
                    
            except Exception as e:
                print(f"  读取文件失败 {sim_dir}: {e}")
                continue
        else:
            print(f"  文件不存在: {sim_dir} (尝试了 results_summary.txt 和 simulation_results_optimized.txt)")
    
    print(f"成功读取 {successful_dirs} 个目录，温度点数量: {len(all_data)}")
    return all_data

def calculate_statistics(all_data):
    """计算统计信息"""
    temps = sorted(all_data.keys())
    
    stats = {
        'temperatures': np.array(temps),
        'avg_energies': [],
        'std_energies': [],
        'avg_magnetizations': [],
        'std_magnetizations': [],
        'avg_specific_heats': [],
        'std_specific_heats': [],
        'avg_susceptibilities': [],
        'std_susceptibilities': [],
        'count': []
    }
    
    for temp in temps:
        data = all_data[temp]
        
        stats['avg_energies'].append(np.mean(data['energies']))
        stats['std_energies'].append(np.std(data['energies']))
        stats['avg_magnetizations'].append(np.mean(data['magnetizations']))
        stats['std_magnetizations'].append(np.std(data['magnetizations']))
        stats['avg_specific_heats'].append(np.mean(data['specific_heats']))
        stats['std_specific_heats'].append(np.std(data['specific_heats']))
        stats['avg_susceptibilities'].append(np.mean(data['susceptibilities']))
        stats['std_susceptibilities'].append(np.std(data['susceptibilities']))
        stats['count'].append(len(data['energies']))
    
    # 转换为numpy数组
    for key in ['avg_energies', 'std_energies', 'avg_magnetizations', 'std_magnetizations',
                'avg_specific_heats', 'std_specific_heats', 'avg_susceptibilities', 'std_susceptibilities']:
        stats[key] = np.array(stats[key])
    
    return stats

def find_peaks(temps, specific_heats, susceptibilities):
    """找到比热和磁化率的峰值及对应温度"""
    # 找到比热峰值
    c_peak_idx = np.argmax(specific_heats)
    c_peak_temp = temps[c_peak_idx]
    c_peak_value = specific_heats[c_peak_idx]
    
    # 找到磁化率峰值
    chi_peak_idx = np.argmax(susceptibilities)
    chi_peak_temp = temps[chi_peak_idx]
    chi_peak_value = susceptibilities[chi_peak_idx]
    
    return {
        'specific_heat': {'temp': c_peak_temp, 'value': c_peak_value, 'index': c_peak_idx},
        'susceptibility': {'temp': chi_peak_temp, 'value': chi_peak_value, 'index': chi_peak_idx}
    }

def create_analysis_plots(stats, peaks):
    """创建XY模型分析图表"""
    print("正在创建XY模型分析图表...")
    
    temps = stats['temperatures']
    
    # 创建2x2的子图，调整尺寸避免过大
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('2D XY Model - Statistical Analysis Results', fontsize=14, fontweight='bold')
    
    # 1. 能量 vs 温度
    ax1.errorbar(temps, stats['avg_energies'], yerr=stats['std_energies'], 
                 fmt='o-', color='blue', linewidth=2, markersize=4, capsize=3,
                 label='Average Energy ± Std Dev')
    ax1.set_xlabel('Temperature $T$', fontsize=12)
    ax1.set_ylabel('Average Energy $\\langle E \\rangle/N$', fontsize=12)
    ax1.set_title('Energy vs Temperature', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # 2. 磁化强度 vs 温度
    ax2.errorbar(temps, np.abs(stats['avg_magnetizations']), yerr=stats['std_magnetizations'], 
                 fmt='o-', color='red', linewidth=2, markersize=4, capsize=3,
                 label='$|\\langle M \\rangle|/N$ ± Std Dev')
    ax2.set_xlabel('Temperature $T$', fontsize=12)
    ax2.set_ylabel('Magnetization $|\\langle M \\rangle|/N$', fontsize=12)
    ax2.set_title('Magnetization vs Temperature', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    
    # 3. 比热 vs 温度 (带误差和峰值标注)
    ax3.errorbar(temps, stats['avg_specific_heats'], yerr=stats['std_specific_heats'], 
                 fmt='o-', color='green', linewidth=2, markersize=4, capsize=3,
                 label='Specific Heat ± Std Dev')
    # 标记峰值
    ax3.plot(peaks['specific_heat']['temp'], peaks['specific_heat']['value'], 
             'r*', markersize=15, label='Peak')
    ax3.axvline(x=peaks['specific_heat']['temp'], color='red', linestyle='--', alpha=0.5)
    
    # 添加简洁的峰值标注 - 放到右上角
    ax3.text(0.98, 0.98, f'$T_c^C$ = {peaks["specific_heat"]["temp"]:.3f}\\n$C_{{max}}$ = {peaks["specific_heat"]["value"]:.4f}',
             transform=ax3.transAxes, fontsize=9, color='red',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
             verticalalignment='top', horizontalalignment='right')
    
    ax3.set_xlabel('Temperature $T$', fontsize=12)
    ax3.set_ylabel('Specific Heat $C$', fontsize=12)
    ax3.set_title('Specific Heat vs Temperature', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # 4. 磁化率 vs 温度 (带误差和峰值标注)
    ax4.errorbar(temps, stats['avg_susceptibilities'], yerr=stats['std_susceptibilities'], 
                 fmt='o-', color='purple', linewidth=2, markersize=4, capsize=3,
                 label='Susceptibility ± Std Dev')
    # 标记峰值
    ax4.plot(peaks['susceptibility']['temp'], peaks['susceptibility']['value'], 
             'r*', markersize=15, label='Peak')
    ax4.axvline(x=peaks['susceptibility']['temp'], color='red', linestyle='--', alpha=0.5)
    
    # 添加简洁的峰值标注 - 放到右上角
    ax4.text(0.98, 0.98, f'$T_c^\\chi$ = {peaks["susceptibility"]["temp"]:.3f}\\n$\\chi_{{max}}$ = {peaks["susceptibility"]["value"]:.4f}',
             transform=ax4.transAxes, fontsize=9, color='red',
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
             verticalalignment='top', horizontalalignment='right')
    
    ax4.set_xlabel('Temperature $T$', fontsize=12)
    ax4.set_ylabel('Susceptibility $\\chi$', fontsize=12)
    ax4.set_title('Susceptibility vs Temperature', fontsize=14, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout(pad=1.5)
    
    # 保存图片
    plt.savefig('xy_model_analysis.png', dpi=100, bbox_inches='tight')
    print("XY model analysis plot saved as: xy_model_analysis.png")
    
    plt.show()

def print_analysis_summary(stats, peaks):
    """打印XY模型分析结果摘要"""
    print("\n" + "="*80)
    print("二维XY模型 - 详细分析结果")
    print("="*80)
    
    print(f"\n数据统计:")
    print(f"  • 温度点数量: {len(stats['temperatures'])}")
    print(f"  • 温度范围: {stats['temperatures'][0]:.3f} - {stats['temperatures'][-1]:.3f}")
    print(f"  • 每个温度点的平均样本数: {np.mean(stats['count']):.1f}")
    
    print(f"\n比热分析:")
    print(f"  • 峰值温度 $T_c^C$ = {peaks['specific_heat']['temp']:.4f}")
    print(f"  • 峰值比热 $C_{{max}}$ = {peaks['specific_heat']['value']:.6f}")
    print(f"  • 峰值处标准差 = {stats['std_specific_heats'][peaks['specific_heat']['index']]:.6f}")
    
    print(f"\n磁化率分析:")
    print(f"  • 峰值温度 $T_c^\\chi$ = {peaks['susceptibility']['temp']:.4f}")
    print(f"  • 峰值磁化率 $\\chi_{{max}}$ = {peaks['susceptibility']['value']:.6f}")
    print(f"  • 峰值处标准差 = {stats['std_susceptibilities'][peaks['susceptibility']['index']]:.6f}")
    
    # 估计临界温度
    tc_from_specific_heat = peaks['specific_heat']['temp']
    tc_from_susceptibility = peaks['susceptibility']['temp']
    tc_average = (tc_from_specific_heat + tc_from_susceptibility) / 2
    
    print(f"\n临界温度估计:")
    print(f"  • 从比热峰值: $T_c = {tc_from_specific_heat:.4f}$")
    print(f"  • 从磁化率峰值: $T_c = {tc_from_susceptibility:.4f}$")
    print(f"  • 平均值: $T_c = {tc_average:.4f}$")
    
    print(f"\n理论背景 (二维XY模型):")
    print(f"  • 二维XY模型在有限温度下没有真正的长程序")
    print("  • 存在Kosterlitz-Thouless相变 (理论值 $T_{KT} ≈ 0.89$)")
    print(f"  • 有限尺寸效应会导致峰值位置和高度发生变化")
    print(f"  • 有限尺寸系统的行为与理论预测仍存在差异")
    
    print("="*80)

def organize_images():
    """组织图片文件"""
    print("\n" + "="*50)
    print("第一步：创建文件夹结构")
    print("="*50)
    
    # 创建温度文件夹
    temperatures = create_test_folders()
    
    # 复制图片
    import shutil
    total_copied = copy_all_images()
    
    print(f"图片组织完成：{total_copied} 张图片已复制到对应温度文件夹")

def analyze_data():
    """分析数据"""
    print("\n" + "="*50)
    print("第二步：数据分析与可视化")
    print("="*50)
    
    # 读取数据
    all_data = read_simulation_data()
    
    if not all_data:
        print("错误: 没有读取到任何数据!")
        return None, None
    
    # 计算统计信息
    stats = calculate_statistics(all_data)
    
    # 找到峰值
    peaks = find_peaks(stats['temperatures'], stats['avg_specific_heats'], stats['avg_susceptibilities'])
    
    # 创建图表
    create_analysis_plots(stats, peaks)
    
    # 打印摘要
    print_analysis_summary(stats, peaks)
    
    return stats, peaks

def main():
    """主函数"""
    print("二维XY模型 - 完整分析程序")
    print("="*60)
    
    while True:
        print("\n请选择要执行的操作:")
        print("1. 组织图片文件（创建温度文件夹并复制图片）")
        print("2. 分析数据并生成图表")
        print("3. 执行完整流程（先组织文件，再分析数据）")
        print("4. 退出")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == '1':
            organize_images()
        elif choice == '2':
            analyze_data()
        elif choice == '3':
            organize_images()
            analyze_data()
        elif choice == '4':
            print("程序结束")
            break
        else:
            print("无效选择，请重新输入")

if __name__ == "__main__":
    main()