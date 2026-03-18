import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import pandas as pd

# 设置字体（根据绘图标准：Source Serif Variable）
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Source Serif Variable', 'Times New Roman', 'DejaVu Serif']
rcParams['axes.unicode_minus'] = False
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42

# 设置绘图标准（基于双栏600pt宽度）
# 双栏图：600pt ≈ 8英寸 (72pt/inch)
DPI = 100
DOUBLE_COL_WIDTH = 8.0  # 英寸
SINGLE_COL_WIDTH = 4.0  # 英寸

# 字体字号规范
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 13
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 13
rcParams['lines.linewidth'] = 1.5
rcParams['grid.linewidth'] = 0.5
rcParams['grid.alpha'] = 0.4
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8





def read_simulation_data():
    """读取4个尺度（8x8, 16x16, 24x24, 32x32）的仿真数据"""
    print("正在读取4个尺度的仿真数据...")

    # 定义4个尺度
    scales = ['8x8', '16x16', '24x24', '32x32']
    base_dir = os.path.join(os.getcwd(), '数据')

    # 存储每个尺度的数据
    scales_data = {}

    for scale in scales:
        scale_dir = os.path.join(base_dir, scale)
        print(f"\n处理尺度: {scale}")
        print(f"  数据目录: {scale_dir}")

        if not os.path.exists(scale_dir):
            print(f"  警告: 目录不存在，跳过")
            continue

        # 读取该尺度下所有仿真结果
        sim_result_files = glob.glob(os.path.join(scale_dir, '相变', 'simulation_results_*', 'simulation_results_optimized.txt'))
        print(f"  找到 {len(sim_result_files)} 个仿真结果文件")

        if not sim_result_files:
            print(f"  警告: 没有找到仿真结果文件")
            continue

        # 收集所有数据
        all_temperatures = []
        all_energies = []
        all_specific_heats = []
        all_magnetizations = []
        all_susceptibilities = []

        successful_reads = 0
        for result_file in sim_result_files:
            try:
                data = np.loadtxt(result_file, encoding='utf-8')

                # 确保数据格式正确
                if data.shape[1] >= 5:
                    temperatures = data[:, 0]
                    energies = data[:, 1]
                    specific_heats = data[:, 2]
                    magnetizations = data[:, 3]
                    susceptibilities = data[:, 4]

                    all_temperatures.extend(temperatures)
                    all_energies.extend(energies)
                    all_specific_heats.extend(specific_heats)
                    all_magnetizations.extend(magnetizations)
                    all_susceptibilities.extend(susceptibilities)

                    successful_reads += 1
                else:
                    print(f"    警告: 数据格式不正确，需要至少5列")

            except Exception as e:
                print(f"    读取文件失败 {os.path.basename(result_file)}: {e}")
                continue

        print(f"  成功读取 {successful_reads}/{len(sim_result_files)} 个文件")

        if successful_reads > 0:
            # 计算统计信息（按温度分组）
            temp_dict = {}
            for i, temp in enumerate(all_temperatures):
                temp_key = round(temp, 6)  # 避免浮点数精度问题
                if temp_key not in temp_dict:
                    temp_dict[temp_key] = {
                        'energies': [],
                        'specific_heats': [],
                        'magnetizations': [],
                        'susceptibilities': []
                    }
                temp_dict[temp_key]['energies'].append(all_energies[i])
                temp_dict[temp_key]['specific_heats'].append(all_specific_heats[i])
                temp_dict[temp_key]['magnetizations'].append(all_magnetizations[i])
                temp_dict[temp_key]['susceptibilities'].append(all_susceptibilities[i])

            # 计算平均值和标准差
            sorted_temps = sorted(temp_dict.keys())
            avg_temps = np.array(sorted_temps)
            avg_energies = np.array([np.mean(temp_dict[t]['energies']) for t in sorted_temps])
            avg_specific_heats = np.array([np.mean(temp_dict[t]['specific_heats']) for t in sorted_temps])
            avg_magnetizations = np.array([np.mean(temp_dict[t]['magnetizations']) for t in sorted_temps])
            avg_susceptibilities = np.array([np.mean(temp_dict[t]['susceptibilities']) for t in sorted_temps])

            std_energies = np.array([np.std(temp_dict[t]['energies']) for t in sorted_temps])
            std_specific_heats = np.array([np.std(temp_dict[t]['specific_heats']) for t in sorted_temps])
            std_magnetizations = np.array([np.std(temp_dict[t]['magnetizations']) for t in sorted_temps])
            std_susceptibilities = np.array([np.std(temp_dict[t]['susceptibilities']) for t in sorted_temps])

            # 存储该尺度的数据
            scales_data[scale] = {
                'temperatures': avg_temps,
                'energies': avg_energies,
                'specific_heats': avg_specific_heats,
                'magnetizations': avg_magnetizations,
                'susceptibilities': avg_susceptibilities,
                'std_energies': std_energies,
                'std_specific_heats': std_specific_heats,
                'std_magnetizations': std_magnetizations,
                'std_susceptibilities': std_susceptibilities
            }

            print(f"  温度点数量: {len(avg_temps)}")
            print(f"  温度范围: {avg_temps[0]:.3f} - {avg_temps[-1]:.3f}")

    print(f"\n成功读取 {len(scales_data)} 个尺度的数据")
    return scales_data

def save_csv_files(scales_data, output_dir):
    """将4个尺度的数据保存为CSV文件"""
    print(f"\n正在保存CSV文件到 {output_dir}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")

    csv_count = 0
    for scale_name, data in scales_data.items():
        csv_file = os.path.join(output_dir, f"{scale_name}.csv")

        # 创建DataFrame
        df = pd.DataFrame({
            'Temperature': data['temperatures'],
            'Energy': data['energies'],
            'Energy_Std': data['std_energies'],
            'Specific_Heat': data['specific_heats'],
            'Specific_Heat_Std': data['std_specific_heats'],
            'Magnetization': data['magnetizations'],
            'Magnetization_Std': data['std_magnetizations'],
            'Susceptibility': data['susceptibilities'],
            'Susceptibility_Std': data['std_susceptibilities']
        })

        df.to_csv(csv_file, index=False, float_format='%.6f')
        print(f"  保存: {csv_file}")
        csv_count += 1

    print(f"成功保存 {csv_count} 个CSV文件")

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
    """创建XY模型分析图表 - 符合科研论文绘图标准"""
    print("正在创建XY模型分析图表...")

    temps = stats['temperatures']

    # 创建2x2的子图，使用标准尺寸
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.33, 6.66))

    # 1. 能量 vs 温度 - Nature风格深蓝色
    ax1.errorbar(temps, stats['avg_energies'], yerr=stats['std_energies'],
                 fmt='o', color='#1F77B4', linewidth=1.5, markersize=3, capsize=2,
                 elinewidth=1, label='Energy ± Std Dev')
    ax1.set_xlabel('Temperature $T$', fontsize=11)
    ax1.set_ylabel('$\\langle E \\rangle/N$', fontsize=11)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='none')
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax1.set_xticks(x_ticks)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax1.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # 2. 磁化强度 vs 温度 - Nature风格橙色
    ax2.errorbar(temps, np.abs(stats['avg_magnetizations']), yerr=stats['std_magnetizations'],
                 fmt='o', color='#FF7F0E', linewidth=1.5, markersize=3, capsize=2,
                 elinewidth=1, label='$|\\langle M \\rangle|/N$ ± Std Dev')
    ax2.set_xlabel('Temperature $T$', fontsize=11)
    ax2.set_ylabel('$|\\langle M \\rangle|/N$', fontsize=11)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, edgecolor='none')
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax2.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    # 3. 比热 vs 温度 - Nature风格绿色
    ax3.errorbar(temps, stats['avg_specific_heats'], yerr=stats['std_specific_heats'],
                 fmt='o', color='#2CA02C', linewidth=1.5, markersize=3, capsize=2,
                 elinewidth=1, label='Specific Heat ± Std Dev')
    # 标记峰值 - Nature风格红色
    ax3.plot(peaks['specific_heat']['temp'], peaks['specific_heat']['value'],
             'r*', markersize=10, label='Peak', zorder=5, color='#D62728')
    ax3.axvline(x=peaks['specific_heat']['temp'], color='#D62728', linestyle='--',
                linewidth=1, alpha=0.6)

    # 添加简洁的峰值标注 - 7pt
    ax3.text(0.98, 0.98, f'$T_c^C$ = {peaks["specific_heat"]["temp"]:.3f}\\n$C_{{max}}$ = {peaks["specific_heat"]["value"]:.4f}',
             transform=ax3.transAxes, fontsize=9, color='#D62728',
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor='none'),
             verticalalignment='top', horizontalalignment='right')

    ax3.set_xlabel('Temperature $T$', fontsize=11)
    ax3.set_ylabel('$C$', fontsize=11)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='none')
    ax3.tick_params(axis='both', which='major', labelsize=11)
    ax3.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax3.set_xticks(x_ticks)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax3.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.04))

    # 4. 磁化率 vs 温度
    ax4.errorbar(temps, stats['avg_susceptibilities'], yerr=stats['std_susceptibilities'],
                 fmt='o', color='purple', linewidth=1.5, markersize=3, capsize=2,
                 elinewidth=1, label='Susceptibility ± Std Dev')
    # 标记峰值
    ax4.plot(peaks['susceptibility']['temp'], peaks['susceptibility']['value'],
             'r*', markersize=10, label='Peak', zorder=5)
    ax4.axvline(x=peaks['susceptibility']['temp'], color='red', linestyle='--',
                linewidth=1, alpha=0.6)

    # 添加简洁的峰值标注 - 7pt
    ax4.text(0.98, 0.98, f'$T_c^\\chi$ = {peaks["susceptibility"]["temp"]:.3f}\\n$\\chi_{{max}}$ = {peaks["susceptibility"]["value"]:.4f}',
             transform=ax4.transAxes, fontsize=9, color='red',
             bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor='none'),
             verticalalignment='top', horizontalalignment='right')

    ax4.set_xlabel('Temperature $T$', fontsize=11)
    ax4.set_ylabel('$\\chi$', fontsize=11)
    ax4.legend(loc='upper left', fontsize=9, framealpha=0.9, edgecolor='none')
    ax4.tick_params(axis='both', which='major', labelsize=11)
    ax4.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax4.set_xticks(x_ticks)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax4.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax4.yaxis.set_minor_locator(plt.MultipleLocator(1.0))

    plt.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8)

    return fig

def create_scale_plots(scales_data, output_dir):
    """为每个尺度创建数据图（带误差棒）- 按照bkt_analysis_v2规格"""
    print(f"\n正在为每个尺度创建数据图，保存到 {output_dir}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")

    plot_count = 0
    # Nature期刊配色方案
    nature_colors = {
        'blue': '#1F77B4',   # 经典科学蓝
        'orange': '#FF7F0E', # 互补橙色
        'green': '#2CA02C',  # 深绿色
        'red': '#D62728'     # 科学红
    }

    for idx, (scale_name, data) in enumerate(scales_data.items()):
        # 双栏图规格: 8.0英寸宽 × 6.4英寸高
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.0, 6.4), dpi=DPI)

        # 为每个尺度分配Nature风格颜色
        scale_color_map = {
            '8x8': nature_colors['red'],   # 8x8 改为红色
            '16x16': nature_colors['orange'],
            '24x24': nature_colors['green'],
            '32x32': nature_colors['blue']  # 32x32 改为蓝色
        }
        color = scale_color_map.get(scale_name, nature_colors['blue'])

        # 根据不同尺度设置横坐标范围
        if scale_name == '8x8':
            xticks_range = np.linspace(1.00, 1.70, 5)
            xticks_labels = ['1.00', '1.18', '1.35', '1.52', '1.70']
            # 纵坐标范围
            yticks_ax1 = np.linspace(-0.68, -0.34, 5)
            ylabels_ax1 = ['-0.68', '-0.60', '-0.51', '-0.43', '-0.34']
            yticks_ax2 = np.linspace(0.28, 0.73, 5)
            ylabels_ax2 = ['0.28', '0.39', '0.50', '0.62', '0.73']
            yticks_ax3 = np.linspace(0.13, 0.33, 5)
            ylabels_ax3 = ['0.13', '0.18', '0.23', '0.28', '0.33']
            yticks_ax4 = np.linspace(0.47, 1.34, 5)
            ylabels_ax4 = ['0.47', '0.69', '0.91', '1.13', '1.34']
        elif scale_name == '16x16':
            xticks_range = np.linspace(0.85, 1.45, 5)
            xticks_labels = ['0.85', '1.00', '1.15', '1.30', '1.45']
            # 纵坐标范围
            yticks_ax1 = np.linspace(-0.75, -0.41, 5)
            ylabels_ax1 = ['-0.75', '-0.67', '-0.58', '-0.50', '-0.41']
            yticks_ax2 = np.linspace(0.19, 0.74, 5)
            ylabels_ax2 = ['0.19', '0.33', '0.47', '0.61', '0.74']
            yticks_ax3 = np.linspace(0.18, 0.37, 5)
            ylabels_ax3 = ['0.18', '0.23', '0.28', '0.33', '0.37']
            yticks_ax4 = np.linspace(0.67, 4.61, 5)
            ylabels_ax4 = ['0.67', '1.66', '2.64', '3.63', '4.61']
        elif scale_name == '24x24':
            xticks_range = np.linspace(0.85, 1.45, 5)
            xticks_labels = ['0.85', '1.00', '1.15', '1.30', '1.45']
            # 纵坐标范围
            yticks_ax1 = np.linspace(-0.74, -0.41, 5)
            ylabels_ax1 = ['-0.74', '-0.66', '-0.58', '-0.50', '-0.41']
            yticks_ax2 = np.linspace(0.13, 0.71, 5)
            ylabels_ax2 = ['0.13', '0.28', '0.42', '0.57', '0.71']
            yticks_ax3 = np.linspace(0.18, 0.38, 5)
            ylabels_ax3 = ['0.18', '0.23', '0.28', '0.33', '0.38']
            yticks_ax4 = np.linspace(1.28, 9.51, 5)
            ylabels_ax4 = ['1.28', '3.59', '5.90', '8.21', '9.51']
        elif scale_name == '32x32':
            xticks_range = np.linspace(0.75, 1.35, 5)
            xticks_labels = ['0.75', '0.90', '1.05', '1.20', '1.35']
            # 纵坐标范围
            yticks_ax1 = np.linspace(-0.78, -0.45, 5)
            ylabels_ax1 = ['-0.78', '-0.70', '-0.63', '-0.55', '-0.45']
            yticks_ax2 = np.linspace(0.11, 0.74, 5)
            ylabels_ax2 = ['0.11', '0.27', '0.42', '0.58', '0.74']
            yticks_ax3 = np.linspace(0.18, 0.38, 5)
            ylabels_ax3 = ['0.18', '0.23', '0.28', '0.33', '0.38']
            yticks_ax4 = np.linspace(1.56, 15.70, 5)
            ylabels_ax4 = ['1.56', '5.50', '9.43', '13.36', '15.70']
        else:
            xticks_range = np.linspace(1.00, 1.50, 5)
            xticks_labels = ['1.00', '1.13', '1.25', '1.38', '1.50']
            yticks_ax1 = np.linspace(-0.80, -0.40, 5)
            ylabels_ax1 = ['-0.80', '-0.70', '-0.60', '-0.50', '-0.40']
            yticks_ax2 = np.linspace(0.10, 0.75, 5)
            ylabels_ax2 = ['0.10', '0.26', '0.42', '0.58', '0.75']
            yticks_ax3 = np.linspace(0.18, 0.38, 5)
            ylabels_ax3 = ['0.18', '0.23', '0.28', '0.33', '0.38']
            yticks_ax4 = np.linspace(0.50, 16.00, 5)
            ylabels_ax4 = ['0.50', '4.38', '8.25', '12.13', '16.00']

        # 能量 vs 温度（带误差棒和连接线）
        ax1.errorbar(data['temperatures'], data['energies'], yerr=data['std_energies'],
                     fmt='o-', color=color, linewidth=1.2, markersize=3, capsize=2,
                     elinewidth=1, label='Energy ± Std Dev',
                     fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)
        ax1.set_xlabel('$T$', fontsize=11)
        ax1.set_ylabel('$\\langle E \\rangle/N$', fontsize=11)
        # 去掉图例
        # ax1.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
        ax1.tick_params(axis='both', which='major', labelsize=11, direction='in')
        ax1.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
        ax1.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)
        ax1.grid(False)

        # 设置横坐标和纵坐标刻度为4个区间（根据尺度调整）
        ax1.set_xticks(xticks_range)
        ax1.set_xticklabels(xticks_labels)
        ax1.set_yticks(yticks_ax1)
        ax1.set_yticklabels(ylabels_ax1)

        # 添加小刻度（主刻度之间一个，从起始位置对齐）
        x_minor_ticks = []
        y_minor_ticks = []
        for i in range(len(xticks_range) - 1):
            x_minor_ticks.append((xticks_range[i] + xticks_range[i+1]) / 2)
        for i in range(len(yticks_ax1) - 1):
            y_minor_ticks.append((yticks_ax1[i] + yticks_ax1[i+1]) / 2)
        ax1.set_xticks(xticks_range, minor=False)
        ax1.set_xticks(x_minor_ticks, minor=True)
        ax1.set_yticks(yticks_ax1, minor=False)
        ax1.set_yticks(y_minor_ticks, minor=True)

        # 添加坐标轴边框
        for spine in ax1.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.6)

        # 磁化强度 vs 温度（带误差棒和连接线）
        ax2.errorbar(data['temperatures'], np.abs(data['magnetizations']),
                     yerr=data['std_magnetizations'],
                     fmt='o-', color=color, linewidth=1.2, markersize=3, capsize=2,
                     elinewidth=1, label='$|\\langle M \\rangle|/N$ ± Std Dev',
                     fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)
        ax2.set_xlabel('$T$', fontsize=11)
        ax2.set_ylabel('$|\\langle M \\rangle|/N$', fontsize=11)
        # 去掉图例
        # ax2.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
        ax2.tick_params(axis='both', which='major', labelsize=11, direction='in')
        ax2.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
        ax2.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)
        ax2.grid(False)

        # 设置横坐标和纵坐标刻度为4个区间（根据尺度调整）
        ax2.set_xticks(xticks_range)
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticks(yticks_ax2)
        ax2.set_yticklabels(ylabels_ax2)

        # 添加小刻度（主刻度之间一个，从起始位置对齐）
        x_minor_ticks = []
        y_minor_ticks = []
        for i in range(len(xticks_range) - 1):
            x_minor_ticks.append((xticks_range[i] + xticks_range[i+1]) / 2)
        for i in range(len(yticks_ax2) - 1):
            y_minor_ticks.append((yticks_ax2[i] + yticks_ax2[i+1]) / 2)
        ax2.set_xticks(xticks_range, minor=False)
        ax2.set_xticks(x_minor_ticks, minor=True)
        ax2.set_yticks(yticks_ax2, minor=False)
        ax2.set_yticks(y_minor_ticks, minor=True)

        for spine in ax2.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.6)

        # 比热 vs 温度（带误差棒和连接线）
        ax3.errorbar(data['temperatures'], data['specific_heats'],
                     yerr=data['std_specific_heats'],
                     fmt='o-', color=color, linewidth=1.2, markersize=3, capsize=2,
                     elinewidth=1, label='Specific Heat ± Std Dev',
                     fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)
        ax3.set_xlabel('$T$', fontsize=11)
        ax3.set_ylabel('$C$', fontsize=11)
        ax3.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
        ax3.tick_params(axis='both', which='major', labelsize=11, direction='in')
        ax3.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
        ax3.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)
        ax3.grid(False)

        # 设置横坐标和纵坐标刻度为4个区间（根据尺度调整）
        ax3.set_xticks(xticks_range)
        ax3.set_xticklabels(xticks_labels)
        ax3.set_yticks(yticks_ax3)
        ax3.set_yticklabels(ylabels_ax3)

        # 添加小刻度（主刻度之间一个，从起始位置对齐）
        x_minor_ticks = []
        y_minor_ticks = []
        for i in range(len(xticks_range) - 1):
            x_minor_ticks.append((xticks_range[i] + xticks_range[i+1]) / 2)
        for i in range(len(yticks_ax3) - 1):
            y_minor_ticks.append((yticks_ax3[i] + yticks_ax3[i+1]) / 2)
        ax3.set_xticks(xticks_range, minor=False)
        ax3.set_xticks(x_minor_ticks, minor=True)
        ax3.set_yticks(yticks_ax3, minor=False)
        ax3.set_yticks(y_minor_ticks, minor=True)

        for spine in ax3.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.6)

        # 磁化率 vs 温度（带误差棒和连接线）
        ax4.errorbar(data['temperatures'], data['susceptibilities'],
                     yerr=data['std_susceptibilities'],
                     fmt='o-', color=color, linewidth=1.2, markersize=3, capsize=2,
                     elinewidth=1, label='Susceptibility ± Std Dev',
                     fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)
        ax4.set_xlabel('$T$', fontsize=11)
        ax4.set_ylabel('$\\chi$', fontsize=11)
        ax4.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
        ax4.tick_params(axis='both', which='major', labelsize=11, direction='in')
        ax4.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
        ax4.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)
        ax4.grid(False)

        # 设置横坐标和纵坐标刻度为4个区间（根据尺度调整）
        ax4.set_xticks(xticks_range)
        ax4.set_xticklabels(xticks_labels)
        ax4.set_yticks(yticks_ax4)
        ax4.set_yticklabels(ylabels_ax4)

        # 添加小刻度（主刻度之间一个，从起始位置对齐）
        x_minor_ticks = []
        y_minor_ticks = []
        for i in range(len(xticks_range) - 1):
            x_minor_ticks.append((xticks_range[i] + xticks_range[i+1]) / 2)
        for i in range(len(yticks_ax4) - 1):
            y_minor_ticks.append((yticks_ax4[i] + yticks_ax4[i+1]) / 2)
        ax4.set_xticks(xticks_range, minor=False)
        ax4.set_xticks(x_minor_ticks, minor=True)
        ax4.set_yticks(yticks_ax4, minor=False)
        ax4.set_yticks(y_minor_ticks, minor=True)

        for spine in ax4.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.6)

        # 去掉标题
        plt.tight_layout(pad=0.2, w_pad=0.1)

        # 保存图片 (同时保存PDF和TIFF)
        plot_file_pdf = os.path.join(output_dir, f"{scale_name}.pdf")
        plot_file_tiff = os.path.join(output_dir, f"{scale_name}.tiff")
        plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        plt.savefig(plot_file_tiff, format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  保存: {plot_file_pdf}")
        print(f"  保存: {plot_file_tiff}")
        plot_count += 1

    print(f"成功生成 {plot_count} 个尺度的数据图")

def create_combined_plot(scales_data, output_dir):
    """创建包含4个尺度的对比图 - 按照bkt_analysis_v2规格"""
    print(f"\n正在创建尺度对比图，保存到 {output_dir}...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 双栏图规格: 8.0英寸宽 × 6.4英寸高
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.0, 6.4), dpi=DPI)

    # Nature期刊配色方案 - 用于尺度对比图
    nature_colors = {
        'blue': '#1F77B4',   # 经典科学蓝
        'orange': '#FF7F0E', # 互补橙色
        'green': '#2CA02C',  # 深绿色
        'red': '#D62728'     # 科学红
    }

    # 为每个尺度分配Nature风格颜色
    scale_color_map = {
        '8x8': nature_colors['red'],   # 8x8 改为红色
        '16x16': nature_colors['orange'],
        '24x24': nature_colors['green'],
        '32x32': nature_colors['blue']  # 32x32 改为蓝色
    }

    # 在同一个图上绘制所有尺度的数据
    for idx, (scale_name, data) in enumerate(scales_data.items()):
        color = scale_color_map.get(scale_name, nature_colors['blue'])

        # 能量
        ax1.plot(data['temperatures'], data['energies'],
                 'o-', color=color, linewidth=1.2, markersize=2, label=scale_name,
                 fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)

        # 磁化强度
        ax2.plot(data['temperatures'], np.abs(data['magnetizations']),
                 'o-', color=color, linewidth=1.2, markersize=2, label=scale_name,
                 fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)

        # 比热
        ax3.plot(data['temperatures'], data['specific_heats'],
                 'o-', color=color, linewidth=1.2, markersize=2, label=scale_name,
                 fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)

        # 磁化率
        ax4.plot(data['temperatures'], data['susceptibilities'],
                 'o-', color=color, linewidth=1.2, markersize=2, label=scale_name,
                 fillstyle='none', markeredgecolor=color, markeredgewidth=0.8)

    # 设置标签和图例
    ax1.set_xlabel('$T$', fontsize=11)
    ax1.set_ylabel('$\\langle E \\rangle/N$', fontsize=11)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    ax1.tick_params(axis='both', which='major', labelsize=11, direction='in')
    ax1.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    ax1.grid(False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax1.set_xticks(x_ticks)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax1.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    for spine in ax1.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    ax2.set_xlabel('$T$', fontsize=11)
    ax2.set_ylabel('$|\\langle M \\rangle|/N$', fontsize=11)
    ax2.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    ax2.tick_params(axis='both', which='major', labelsize=11, direction='in')
    ax2.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    ax2.grid(False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax2.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax2.yaxis.set_minor_locator(plt.MultipleLocator(0.1))

    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    ax3.set_xlabel('$T$', fontsize=11)
    ax3.set_ylabel('$C$', fontsize=11)
    ax3.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    ax3.tick_params(axis='both', which='major', labelsize=11, direction='in')
    ax3.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    ax3.grid(False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax3.set_xticks(x_ticks)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax3.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.04))

    for spine in ax3.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    ax4.set_xlabel('$T$', fontsize=11)
    ax4.set_ylabel('$\\chi$', fontsize=11)
    ax4.legend(loc='best', fontsize=9, framealpha=0.9, frameon=False)
    ax4.tick_params(axis='both', which='major', labelsize=11, direction='in')
    ax4.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)
    ax4.grid(False)
    x_ticks = np.linspace(0.88, 1.14, 5)
    ax4.set_xticks(x_ticks)
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
    # 添加小刻度在主刻度之间
    x_minor = [(x_ticks[i] + x_ticks[i+1]) / 2 for i in range(len(x_ticks) - 1)]
    ax4.set_xticks(x_minor, minor=True)
    # 纵坐标小刻度
    ax4.yaxis.set_minor_locator(plt.MultipleLocator(1.0))

    for spine in ax4.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    plt.tight_layout(pad=0.2, w_pad=0.1)

    # 保存图片 (同时保存PDF和TIFF)
    plot_file_pdf = os.path.join(output_dir, "combined_scales_comparison.pdf")
    plot_file_tiff = os.path.join(output_dir, "combined_scales_comparison.tiff")
    plt.savefig(plot_file_pdf, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(plot_file_tiff, format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  保存: {plot_file_pdf}")
    print(f"  保存: {plot_file_tiff}")

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

def analyze_data():
    """分析数据"""
    print("\n" + "="*50)
    print("数据分析与可视化")
    print("="*50)

    # 读取数据
    scales_data = read_simulation_data()

    if not scales_data:
        print("错误: 没有读取到任何数据!")
        return None, None

    # 保存CSV文件
    csv_output_dir = "csv_data"
    save_csv_files(scales_data, csv_output_dir)

    # 生成每个尺度的数据图
    plots_output_dir = "scale_plots"
    create_scale_plots(scales_data, plots_output_dir)

    # 生成尺度对比图
    create_combined_plot(scales_data, plots_output_dir)

    print("\n所有数据处理完成！")
    print(f"- CSV文件保存在: {csv_output_dir}")
    print(f"- 数据图保存在: {plots_output_dir}")

    return scales_data

def main():
    """主函数"""
    print("二维XY模型 - 多尺度数据分析与可视化程序")
    print("="*60)

    # 直接执行数据分析
    analyze_data()

if __name__ == "__main__":
    main()