import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import matplotlib
import os

# 简单设置 - 直接使用英文，避免字体问题
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 设置专业期刊风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['lines.linewidth'] = 1.8

# ========== 函数：读取磁化率数据 ==========
def read_susceptibility_data(filepath):
    """
    Read susceptibility data file

    Parameters:
        filepath: File path

    Returns:
        temps: Temperature array
        chi: Susceptibility array (multiplied by 10^-3)
        chi_std: Susceptibility standard deviation array
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过注释行
    data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]

    # 解析数据
    temps = []
    chi = []
    chi_std = []

    for line in data_lines:
        parts = line.split()
        if len(parts) >= 5:
            temps.append(float(parts[0]))
            # 第4列是磁化率，第5列是标准差
            chi.append(float(parts[3]) * 1e-3)  # 转换回实际值
            chi_std.append(float(parts[4]) * 1e-3)

    return np.array(temps), np.array(chi), np.array(chi_std)

# ========== 读取所有数据 ==========
data_dir = 'd:/LX/绘图数据/传统序参量'
sizes = [16, 32, 64, 128, 256]

data_dict = {}

for L in sizes:
    filepath = os.path.join(data_dir, f'{L}_susceptibility_results.txt')
    temps, chi, chi_std = read_susceptibility_data(filepath)

    data_dict[L] = {
        'temps': temps,
        'chi': chi,
        'chi_std': chi_std
    }

    print(f"L={L}: 读取了 {len(temps)} 个温度点，温度范围 [{temps.min():.3f}, {temps.max():.3f}] K")

# ========== CNN预测的T_c(L) ==========
T_cnn_dict = {
    16: 1.142000,
    32: 1.039017,
    64: 1.004737,
    128: 0.973900,  # Bootstrap均值：0.9742 K，Bootstrap分布：0.9739 ± 0.0026 K
    256: 0.971898
}

print("\nCNN预测的T_c(L):")
for L in sizes:
    print(f"  L={L:3d}: {T_cnn_dict[L]:.6f} K")

# ========== 找到磁化率峰值 ==========
peak_temps = []
peak_chis = []
peak_chi_stds = []

print("\nSusceptibility Peaks:")
for L in sizes:
    temps = data_dict[L]['temps']
    chi = data_dict[L]['chi']
    chi_std = data_dict[L]['chi_std']

    # Find peak position
    peak_idx = np.argmax(chi)
    peak_temp = temps[peak_idx]
    peak_chi_val = chi[peak_idx]
    peak_chi_std_val = chi_std[peak_idx]

    peak_temps.append(peak_temp)
    peak_chis.append(peak_chi_val)
    peak_chi_stds.append(peak_chi_std_val)

    print(f"  L={L:3d}: T_peak = {peak_temp:.6f} K, χ_peak = {peak_chi_val:.6f} ± {peak_chi_std_val:.6f}")

# ========== 开始画图 ==========
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# 定义颜色（专业期刊风格）
colors = ['#E64B35', '#4DBBD5', '#00A087', '#F39B7F', '#3C5488']
markers = ['o', 's', '^', 'D', 'v']
markers_peak = ['*', '*', '*', '*', '*']

# 遍历每个尺寸，画磁化率曲线
for idx, L in enumerate(sizes):
    data = data_dict[L]
    temps = data['temps']
    chi = data['chi']
    chi_std = data['chi_std']
    t_cnn = T_cnn_dict[L]

    # 画磁化率曲线
    ax.plot(temps, chi,
           color=colors[idx],
           linewidth=2.0,
           linestyle='-',
           label=f'L={L}',
           zorder=2,
           alpha=0.9)

    # 画数据点（每隔几个点画一个，避免太密）
    step = max(1, len(temps) // 20)
    ax.scatter(temps[::step], chi[::step],
              color=colors[idx],
              marker=markers[idx],
              s=50,
              alpha=0.5,
              zorder=3)

    # Mark peak position
    peak_temp = peak_temps[idx]
    peak_chi = peak_chis[idx]
    ax.plot(peak_temp, peak_chi,
           marker='*',
           markersize=20,
           color=colors[idx],
           markeredgecolor='black',
           markeredgewidth=1.5,
           zorder=6,
           label=f'L={L} Peak' if idx == 0 else '')

    # 标记CNN预测的T_c(L)位置（空心菱形）
    if min(temps) <= t_cnn <= max(temps):
        chi_at_cnn = np.interp(t_cnn, temps, chi)
        ax.plot(t_cnn, chi_at_cnn,
               marker='D',
               markersize=12,
               color=colors[idx],
               markeredgecolor='black',
               markeredgewidth=2.0,
               fillstyle='none',
               zorder=5,
               label=f'L={L} CNN' if idx == 0 else '')
    else:
        print(f"Warning: L={L} CNN temp {t_cnn:.3f} out of range [{temps.min():.3f}, {temps.max():.3f}]")

# ========== 设置坐标轴标签 ==========
ax.set_xlabel('Temperature $T$ (K)', fontsize=15, fontweight='bold', family='serif')
ax.set_ylabel('Magnetic Susceptibility $\\chi$ (log scale)', fontsize=15, fontweight='bold', family='serif')
ax.set_title('Magnetic Susceptibility Peak vs CNN Prediction (Log Scale)', fontsize=17, fontweight='bold', pad=15, family='serif')

# ========== 设置坐标轴范围 ==========
all_temps = np.concatenate([data_dict[L]['temps'] for L in sizes])
all_chi = np.concatenate([data_dict[L]['chi'] for L in sizes])

ax.set_xlim(all_temps.min() - 0.01, all_temps.max() + 0.01)

# 使用对数坐标解决数据量级差异过大的问题
ax.set_yscale('log')
# 设置合适的对数范围，避免log(0)
min_chi = min([data['chi'][data['chi'] > 0].min() for data in data_dict.values()])
ax.set_ylim(min_chi * 0.5, max(all_chi) * 1.5)

# ========== 添加网格 ==========
ax.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)

# ========== 添加图例 ==========
legend = ax.legend(loc='upper left',
                   fontsize=10,
                   ncol=1,
                   frameon=True,
                   fancybox=True,
                   shadow=False,
                   borderpad=0.5,
                   columnspacing=1.0)

# ========== 计算并显示偏差 ==========
print("\nSusceptibility Peak vs CNN Prediction Difference:")
for idx, L in enumerate(sizes):
    peak_temp = peak_temps[idx]
    t_cnn = T_cnn_dict[L]
    diff = t_cnn - peak_temp
    rel_diff = 100 * diff / peak_temp
    print(f"  L={L:3d}: ΔT = {diff:+.6f} K ({rel_diff:+.3f}%)")

# ========== 调整布局 ==========
plt.tight_layout(pad=0.5)

# ========== 保存图片 ==========
output_dir = 'd:/LX/绘图数据'
output_png = os.path.join(output_dir, 'magnetization_comparison.png')
output_pdf = os.path.join(output_dir, 'magnetization_comparison.pdf')

plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')

print(f"\nSaved Figure 1:")
print(f"  PNG: {output_png}")
print(f"  PDF: {output_pdf}")

plt.close()

# ========== 额外画一张图：只显示峰值和CNN预测的对比 ==========
fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=100)

# 画磁化率峰值
ax2.errorbar(sizes, peak_temps,
            fmt='o-', color='#E64B35', linewidth=2.5, markersize=10,
            capsize=5, capthick=1.5, label='$\\chi$ Peak $T_{peak}(L)$')

# 画CNN预测
ax2.plot(sizes, [T_cnn_dict[L] for L in sizes],
        's--', color='#00A087', linewidth=2.5, markersize=10,
        label='CNN Prediction $T_c(L)$')

# 设置坐标轴
ax2.set_xlabel('System Size $L$', fontsize=15, fontweight='bold', family='serif')
ax2.set_ylabel('Temperature $T$ (K)', fontsize=15, fontweight='bold', family='serif')
ax2.set_title('Finite Size Scaling: Peak vs CNN Prediction', fontsize=17, fontweight='bold', pad=15, family='serif')

# 设置x轴为对数刻度
ax2.set_xscale('log')
ax2.set_xticks(sizes)
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

# 添加网格
ax2.grid(True, linestyle=':', alpha=0.4, linewidth=0.8)

# 添加图例
ax2.legend(fontsize=12, loc='lower left')

# 计算相对误差
relative_errors = [100 * (T_cnn_dict[L] - peak_temps[idx]) / peak_temps[idx]
                  for idx, L in enumerate(sizes)]

# 在图上标注偏差
for idx, L in enumerate(sizes):
    ax2.annotate(f'{relative_errors[idx]:+.2f}%',
                xy=(L, T_cnn_dict[L]),
                xytext=(L * 1.15, T_cnn_dict[L]),
                fontsize=9,
                ha='left',
                va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()

# 保存图片2
output_png2 = os.path.join(output_dir, 'magnetization_peak_vs_cnn.png')
output_pdf2 = os.path.join(output_dir, 'magnetization_peak_vs_cnn.pdf')

plt.savefig(output_png2, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_pdf2, format='pdf', bbox_inches='tight', facecolor='white')

print(f"\nSaved Figure 2:")
print(f"  PNG: {output_png2}")
print(f"  PDF: {output_pdf2}")

plt.close()
