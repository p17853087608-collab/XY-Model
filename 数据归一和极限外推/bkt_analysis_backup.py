"""
BKT相变温度分析 
对每个温度点执行Bootstrap,计算均值和标准差
从Bootstrap均值曲线找50%概率点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os
import warnings
warnings.filterwarnings('ignore')

# 设置字体（使用英文，避免字体问题）
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 设置科学期刊风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 16
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['grid.alpha'] = 0.4

# ============== 配置部分 ==============
# 定义文件夹和对应的CSV文件路径
data_info = {
    '16': {
        'path': 'd:/LX/绘图数据/16x16/overall_summary.csv',
        'L': 16,
        'raw_data_path': 'd:/LX/绘图数据/16x16'
    },
    '32': {
        'path': 'd:/LX/绘图数据/32x32/overall_summary.csv',
        'L': 32,
        'raw_data_path': 'd:/LX/绘图数据/32x32'
    },
    '64': {
        'path': 'd:/LX/绘图数据/64x64/overall_summary.csv',
        'L': 64,
        'raw_data_path': 'd:/LX/绘图数据/64x64'
    },
    '128': {
        'path': 'd:/LX/绘图数据/128x128/overall_summary.csv',
        'L': 128,
        'raw_data_path': 'd:/LX/绘图数据/128x128'
    },
    '256': {
        'path': 'd:/LX/绘图数据/256x256/overall_summary.csv',
        'L': 256,
        'raw_data_path': 'd:/LX/绘图数据/256x256'
    }
}

# Bootstrap参数
N_BOOTSTRAP = 1000  # Bootstrap次数

# 输出目录
OUTPUT_DIR = 'd:/LX/绘图数据/bkt_analysis_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============== 数据加载函数 ==============
def load_summary_data(csv_path):
    """加载overall_summary.csv文件"""
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {
            'temp': df['folder_name'].values,
            'ordered_prob': df['avg_ordered_probability'].values,
            'ordered_var': df['var_ordered_probability'].values,
            'amorphous_prob': df['avg_amorphous_probability'].values,
            'amorphous_var': df['var_amorphous_probability'].values
        }
    else:
        print(f"警告: 文件不存在 {csv_path}")
        return None

def load_raw_probability_data(folder_path, temps):
    """加载每个温度点的原始概率数据(有序相和无序相)"""
    raw_probs_ordered = {}
    raw_probs_amorphous = {}
    for temp in temps:
        prob_file = os.path.join(folder_path, 'save_data', f'{temp:.3f}.csv')
        if os.path.exists(prob_file):
            df = pd.read_csv(prob_file)
            raw_probs_ordered[temp] = df['ordered_probability'].values
            raw_probs_amorphous[temp] = df['amorphous_probability'].values
    return raw_probs_ordered, raw_probs_amorphous

# ============== Bootstrap分析函数 ==============
def bootstrap_probability_curves(temps, raw_probs_ordered, raw_probs_amorphous, n_bootstrap=1000):
    """
    对每个温度点执行Bootstrap重采样
    返回:
        - prob_mean_ordered: 有序相Bootstrap均值
        - prob_std_ordered: 有序相Bootstrap标准差
        - prob_mean_amorphous: 无序相Bootstrap均值
        - prob_std_amorphous: 无序相Bootstrap标准差
        - tc_bootstrap: 1000个T_c(L)估计值
    """
    prob_mean_ordered = np.zeros(len(temps))
    prob_std_ordered = np.zeros(len(temps))
    prob_mean_amorphous = np.zeros(len(temps))
    prob_std_amorphous = np.zeros(len(temps))
    tc_bootstrap = []

    print(f"  执行 {n_bootstrap} 次Bootstrap...")

    for i in range(n_bootstrap):
        bootstrap_probs_ordered = []
        bootstrap_probs_amorphous = []
        for temp in temps:
            if temp in raw_probs_ordered and len(raw_probs_ordered[temp]) > 0:
                # 有放回抽取有序相
                resampled_ordered = np.random.choice(raw_probs_ordered[temp],
                                                   size=len(raw_probs_ordered[temp]), replace=True)
                bootstrap_probs_ordered.append(np.mean(resampled_ordered))

                # 有放回抽取无序相
                resampled_amorphous = np.random.choice(raw_probs_amorphous[temp],
                                                     size=len(raw_probs_amorphous[temp]), replace=True)
                bootstrap_probs_amorphous.append(np.mean(resampled_amorphous))
            else:
                bootstrap_probs_ordered.append(np.nan)
                bootstrap_probs_amorphous.append(np.nan)

        bootstrap_probs_ordered = np.array(bootstrap_probs_ordered)
        bootstrap_probs_amorphous = np.array(bootstrap_probs_amorphous)

        # 累积用于计算均值和标准差
        prob_mean_ordered = (prob_mean_ordered * i + bootstrap_probs_ordered) / (i + 1)
        prob_mean_amorphous = (prob_mean_amorphous * i + bootstrap_probs_amorphous) / (i + 1)

        if i > 0:
            diff_ordered = bootstrap_probs_ordered - prob_mean_ordered
            diff_amorphous = bootstrap_probs_amorphous - prob_mean_amorphous
            prob_std_ordered = np.sqrt((prob_std_ordered**2 * i + diff_ordered**2 / (i + 1)))
            prob_std_amorphous = np.sqrt((prob_std_amorphous**2 * i + diff_amorphous**2 / (i + 1)))

        # 找到50%概率点(使用有序相概率)
        valid_mask = ~np.isnan(bootstrap_probs_ordered)
        if np.any(valid_mask):
            tc, _ = find_tc_at_50(temps[valid_mask], bootstrap_probs_ordered[valid_mask])
            tc_bootstrap.append(tc)

        if (i + 1) % 200 == 0:
            print(f"    进度: {i+1}/{n_bootstrap}")

    return prob_mean_ordered, prob_std_ordered, prob_mean_amorphous, prob_std_amorphous, np.array(tc_bootstrap)

def find_tc_at_50(temp, prob):
    """使用三次样条插值精确找到50%概率对应的温度"""
    from scipy.interpolate import CubicSpline

    # 检查是否有跨越0.5
    if not (np.min(prob) <= 0.5 <= np.max(prob)):
        # 没有跨越0.5，返回最近的点
        idx = np.argmin(np.abs(prob - 0.5))
        return temp[idx], prob[idx]

    if len(temp) < 4:
        # 数据点太少，使用最近邻
        idx = np.argmin(np.abs(prob - 0.5))
        return temp[idx], prob[idx]

    try:
        # 使用三次样条插值
        spline = CubicSpline(temp, prob)

        # 二分法找50%点
        t_min, t_max = np.min(temp), np.max(temp)
        for _ in range(50):
            t_mid = (t_min + t_max) / 2
            p_mid = spline(t_mid)

            if np.isnan(p_mid):
                break

            if prob[0] > prob[-1]:  # 下降曲线
                if p_mid > 0.5:
                    t_min = t_mid
                else:
                    t_max = t_mid
            else:  # 上升曲线
                if p_mid < 0.5:
                    t_min = t_mid
                else:
                    t_max = t_mid

        tc = (t_min + t_max) / 2
        return tc, spline(tc)
    except:
        # 插值失败，回退到最近邻
        idx = np.argmin(np.abs(prob - 0.5))
        return temp[idx], prob[idx]

# ============== BKT拟合函数 ==============
def bkt_scale_fit(tc_mean, tc_std, sizes):
    """BKT标度律拟合: T_c(L) = T_BKT + A / (ln L)^2"""
    x = 1.0 / (np.log(sizes.astype(float)) ** 2)

    def linear_func(x, T_BKT, A):
        return T_BKT + A * x

    valid_mask = ~np.isnan(tc_mean) & ~np.isnan(tc_std)
    x_valid = x[valid_mask]
    y_valid = tc_mean[valid_mask]
    w_valid = 1.0 / (tc_std[valid_mask] ** 2)

    if len(x_valid) > 2:
        popt, pcov = curve_fit(linear_func, x_valid, y_valid, p0=[1.0, 0.1], sigma=1/np.sqrt(w_valid))
        T_BKT, A = popt
        T_BKT_err = np.sqrt(np.diag(pcov))[0]
        return T_BKT, A, T_BKT_err, x_valid, y_valid
    else:
        return None, None, None, None, None

# ============== 数据变换函数 ==============
def transform_probability_curve(temps, probs, tc_L, T_BKT):
    """对概率曲线进行标度变换: T' = T - [T_c(L) - T_BKT]"""
    shift = tc_L - T_BKT
    transformed_temps = temps - shift
    return transformed_temps, probs

# ============== 绘图函数 ==============
def plot_figure_2a(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means):
    """
    Figure 2a: Bootstrap mean probability curves with error bars - ordered and amorphous phases
    MLST journal style
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    size_order = ['16', '32', '64', '128', '256']
    # 简化配色方案：使用暖色调表示有序相，冷色调表示无序相
    # 每个尺寸使用相同色系的不同深浅
    colors_ordered = ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F']  # 红色系（有序相）
    colors_amorphous = ['#1976D2', '#42A5F5', '#66BB6A', '#26A69A', '#4DB6AC']  # 蓝色系（无序相）
    markers = ['o', 's', '^', 'D', 'p']
    linestyles = ['-', '-', '-', '-', '-']

    # Ordered phase curves
    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_ordered_dict[size_key]
        probs_std = prob_std_ordered_dict[size_key]
        tc = tc_from_means[size_key]

        ax.errorbar(temps, probs_mean, yerr=probs_std,
                   color=colors_ordered[idx], marker=markers[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle=linestyles[idx], alpha=0.85,
                   label=f'L={L} (Ordered)', fillstyle='none',
                   markeredgecolor='darkred', markeredgewidth=1.0,
                   zorder=3)

        # Mark 50% point
        ax.plot(tc, 0.5, marker='*', markersize=18,
               color=colors_ordered[idx], markeredgecolor='black',
               markeredgewidth=1.0, zorder=10, fillstyle='none')

    # Amorphous phase curves
    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_amorphous_dict[size_key]
        probs_std = prob_std_amorphous_dict[size_key]

        ax.errorbar(temps, probs_mean, yerr=probs_std,
                   color=colors_amorphous[idx], marker=markers[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='--', alpha=0.85, dash_capstyle='round',
                   label=f'L={L} (Amorphous)', fillstyle='none',
                   markeredgecolor='darkblue', markeredgewidth=1.0,
                   zorder=2)

    ax.axhline(y=0.5, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax.set_xlabel('Temperature $T$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='gray')

    # 分开图例：有序相和无序相分成两列
    legend = ax.legend(loc='upper right', fontsize=8, ncol=2,
                     frameon=True, framealpha=0.95, fancybox=True,
                     borderpad=0.5, handlelength=2.2, columnspacing=1.2)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, 'figure_2a_bootstrap_means.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2a_bootstrap_means.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_2b(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means, T_BKT):
    """
    Figure 2b: BKT scaled probability curves - ordered and amorphous phases
    MLST journal style
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    size_order = ['16', '32', '64', '128', '256']
    # 与Figure 2a相同的配色方案
    colors_ordered = ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F']  # 红色系（有序相）
    colors_amorphous = ['#1976D2', '#42A5F5', '#66BB6A', '#26A69A', '#4DB6AC']  # 蓝色系（无序相）
    markers = ['o', 's', '^', 'D', 'p']

    # Ordered phase curves
    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_ordered_dict[size_key]
        probs_std = prob_std_ordered_dict[size_key]
        tc = tc_from_means[size_key]

        transformed_temps, transformed_probs = transform_probability_curve(temps, probs_mean, tc, T_BKT)

        ax.errorbar(transformed_temps, transformed_probs, yerr=probs_std,
                   color=colors_ordered[idx], marker=markers[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='-', alpha=0.85,
                   label=f'L={L} (Ordered)', fillstyle='none',
                   markeredgecolor='darkred', markeredgewidth=1.0,
                   zorder=3)

    # Amorphous phase curves
    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_amorphous_dict[size_key]
        probs_std = prob_std_amorphous_dict[size_key]
        tc = tc_from_means[size_key]

        transformed_temps, transformed_probs = transform_probability_curve(temps, probs_mean, tc, T_BKT)

        ax.errorbar(transformed_temps, transformed_probs, yerr=probs_std,
                   color=colors_amorphous[idx], marker=markers[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='--', alpha=0.85, dash_capstyle='round',
                   label=f'L={L} (Amorphous)', fillstyle='none',
                   markeredgecolor='darkblue', markeredgewidth=1.0,
                   zorder=2)

    ax.axhline(y=0.5, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.axvline(x=T_BKT, color='#D32F2F', linestyle='--', linewidth=1.8, alpha=0.7, zorder=0)

    ax.set_xlabel('Scaled Temperature $T\'$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='gray')

    legend = ax.legend(loc='upper right', fontsize=8, ncol=2,
                     frameon=True, framealpha=0.95, fancybox=True,
                     borderpad=0.5, handlelength=2.2, columnspacing=1.2)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, 'figure_2b_scaled_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2b_scaled_curves.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_3(tc_mean, tc_std, sizes, T_BKT, A, x_fit, y_fit):
    """
    Figure 3: BKT scaling law fitting
    MLST journal style
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=100)

    # Plot data points with error bars
    ax.errorbar(x_fit, y_fit, yerr=tc_std[:len(x_fit)],
               fmt='o', markersize=10, capsize=5,
               linewidth=2, color='#3C5488',
               label='Finite size $T_c(L)$',
               fillstyle='none', markeredgecolor='#3C5488')

    # Fitted line
    x_fit_line = np.linspace(0, max(x_fit) * 1.1, 100)
    y_fit_line = T_BKT + A * x_fit_line
    ax.plot(x_fit_line, y_fit_line, color='#E64B35', linewidth=2.5,
            label=f'$T_c(L) = {T_BKT:.4f} + {A:.4f} / (\\ln L)^2$')

    # Mark T_BKT
    ax.plot(0, T_BKT, marker='*', markersize=22,
           color='#E64B35', markeredgecolor='black',
           markeredgewidth=1, zorder=10, fillstyle='none')

    ax.set_xlabel('$1/(\\ln L)^2$', fontsize=13, fontweight='bold')
    ax.set_ylabel('$T_c(L)$ (K)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)

    legend = ax.legend(loc='best', fontsize=10, frameon=True,
                     framealpha=0.95, fancybox=True, borderpad=0.5)
    legend.get_frame().set_edgecolor('#888888')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.3)

    output_path = os.path.join(OUTPUT_DIR, 'figure_3_bkt_fit.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_bkt_fit.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_4(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict, tc_from_means):
    """
    Figure 4: Zoom-in of error bars near 50% probability point
    Bootstrap standard deviation distribution (ordered phase only)
    MLST journal style
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    size_order = ['16', '32', '64', '128', '256']
    # 只使用红色系（有序相）
    colors = ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F']
    markers = ['o', 's', '^', 'D', 'p']

    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_ordered_dict[size_key]
        probs_std = prob_std_ordered_dict[size_key]
        tc = tc_from_means[size_key]

        # Find temperature range near 50% point (±0.1K)
        mask = (temps >= tc - 0.1) & (temps <= tc + 0.1)
        temps_zoom = temps[mask]
        probs_mean_zoom = probs_mean[mask]
        probs_std_zoom = probs_std[mask]

        # Plot error bars
        ax.errorbar(temps_zoom, probs_mean_zoom, yerr=probs_std_zoom,
                   color=colors[idx], marker=markers[idx],
                   markersize=7, linewidth=2.0, capsize=4, capthick=1.5,
                   linestyle='-', alpha=0.85, elinewidth=2.0,
                   label=f'L={L}, $T_c$={tc:.4f}K',
                   fillstyle='none', markeredgecolor='darkred',
                   markeredgewidth=1.0, zorder=3)

        # Mark 50% point
        ax.plot(tc, 0.5, marker='*', markersize=20,
               color=colors[idx], markeredgecolor='black',
               markeredgewidth=1.0, zorder=10, fillstyle='none')

    ax.axhline(y=0.5, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax.set_xlabel('Temperature $T$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ordered Phase Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0.3, 0.7)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color='gray')

    legend = ax.legend(loc='upper right', fontsize=9, frameon=True,
                     framealpha=0.95, fancybox=True, borderpad=0.5,
                     columnspacing=1.0)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_error_zoom.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_error_zoom.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_5(tc_bootstrap_dict, T_BKT):
    """
    Figure 5: Finite size effect box plot
    MLST journal style - 优化版，标注在右侧，不遮挡箱线图
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    size_order = ['16', '32', '64', '128', '256']
    sizes = []
    data_list = []
    means = []
    medians = []
    stds = []

    for size_key in size_order:
        if size_key not in tc_bootstrap_dict:
            continue
        L = data_info[size_key]['L']
        sizes.append(L)
        data_list.append(tc_bootstrap_dict[size_key])
        means.append(np.mean(tc_bootstrap_dict[size_key]))
        medians.append(np.median(tc_bootstrap_dict[size_key]))
        stds.append(np.std(tc_bootstrap_dict[size_key], ddof=1))

    # MLST style box plot parameters
    boxprops = {'linewidth': 1.8, 'color': '#333333'}
    whiskerprops = {'linewidth': 1.8, 'color': '#555555'}
    capprops = {'linewidth': 1.8, 'color': '#555555'}
    medianprops = {'linewidth': 2.2, 'color': '#3C5488'}
    meanprops = {'marker': 'D', 'markerfacecolor': '#E64B35',
                'markeredgecolor': '#333333', 'markersize': 9, 'markeredgewidth': 1.2}

    box_plot = ax.boxplot(data_list, labels=[f'{L}' for L in sizes],
                        patch_artist=True, showmeans=True, showfliers=False,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        capprops=capprops,
                        medianprops=medianprops,
                        meanprops=meanprops,
                        widths=0.65)

    # Professional color scheme - 红色系渐变
    colors = ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.8)
        patch.set_alpha(0.7)

    # Plot mean line
    ax.plot(range(1, len(sizes)+1), means, 'o-',
           color='#E64B35', linewidth=2, markersize=7,
           markeredgecolor='#333333', markeredgewidth=1.2,
           label='Mean', zorder=5, fillstyle='none')

    # Plot T_BKT reference line
    ax.axhline(y=T_BKT, color='#D32F2F', linestyle='--', linewidth=2,
              alpha=0.7, zorder=4)

    # Add standard deviation annotations - 在右侧添加，不遮挡箱线图
    for i, (L, mean, std) in enumerate(zip(sizes, means, stds)):
        x_pos = i + 1
        y_upper = mean + std
        y_lower = mean - std

        # Plot error range dotted line
        ax.plot([x_pos, x_pos], [y_lower, y_upper],
               color='#D32F2F', linestyle=':', linewidth=1.5, alpha=0.6)

        # 在右侧添加标准差标注，位置根据y值错开
        annotation_x = 5.8
        annotation_y = mean
        color_idx = i % 2  # 交替使用颜色
        text_color = colors[i]

        # 绘制指向线
        ax.plot([x_pos + 0.35, annotation_x], [annotation_y, annotation_y],
               color=text_color, linestyle='-', linewidth=1, alpha=0.5)

        # Add standard deviation annotation - 简洁格式
        ax.text(annotation_x, annotation_y,
               f'L={L}\nσ={std:.4f}',
               ha='left', va='center',
               fontsize=8, fontweight='bold',
               color='#333333')

    ax.set_xlabel('System Size $L$', fontsize=13, fontweight='bold')
    ax.set_ylabel('$T_c(L)$ (K)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, axis='y')

    # Adjust Y-axis range
    all_data = np.concatenate(data_list)
    y_min = np.min(all_data) - 0.005
    y_max = np.max(all_data) + 0.005
    ax.set_ylim(y_min, y_max)

    # 扩展X轴范围以容纳右侧标注
    ax.set_xlim(0.5, 6.5)

    legend = ax.legend(loc='upper left', fontsize=9, frameon=True,
                     framealpha=0.95, fancybox=True, borderpad=0.5)
    legend.get_frame().set_edgecolor('#888888')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.3)

    output_path = os.path.join(OUTPUT_DIR, 'figure_5_boxplot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_5_boxplot.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

# ============== 主流程 ==============
def main():
    print("="*70)
    print("BKT相变温度分析 - 修正版V2")
    print("="*70)

    # 1. 数据加载
    print("\n[步骤1] 加载数据...")
    temps_dict = {}
    raw_probs_ordered_dict = {}
    raw_probs_amorphous_dict = {}

    for size_key, info in data_info.items():
        print(f"  加载尺寸 L={info['L']}...")

        summary_data = load_summary_data(info['path'])
        if summary_data is None:
            continue

        temps_dict[size_key] = summary_data['temp']
        raw_probs_ordered_dict[size_key], raw_probs_amorphous_dict[size_key] = \
            load_raw_probability_data(info['raw_data_path'], summary_data['temp'])

    # 2. Bootstrap分析
    print(f"\n[步骤2] 对每个温度点执行Bootstrap分析 (每个尺寸{N_BOOTSTRAP}次)...")
    prob_mean_ordered_dict = {}
    prob_std_ordered_dict = {}
    prob_mean_amorphous_dict = {}
    prob_std_amorphous_dict = {}
    tc_bootstrap_dict = {}
    tc_from_means = {}

    for size_key, info in data_info.items():
        if size_key not in temps_dict:
            continue

        print(f"  处理尺寸 L={info['L']}...")
        temps = temps_dict[size_key]

        prob_mean_ordered, prob_std_ordered, prob_mean_amorphous, prob_std_amorphous, tc_samples = \
            bootstrap_probability_curves(temps, raw_probs_ordered_dict[size_key],
                                      raw_probs_amorphous_dict[size_key], N_BOOTSTRAP)

        prob_mean_ordered_dict[size_key] = prob_mean_ordered
        prob_std_ordered_dict[size_key] = prob_std_ordered
        prob_mean_amorphous_dict[size_key] = prob_mean_amorphous
        prob_std_amorphous_dict[size_key] = prob_std_amorphous
        tc_bootstrap_dict[size_key] = tc_samples

        # 从Bootstrap均值曲线找50%点
        tc, _ = find_tc_at_50(temps, prob_mean_ordered)
        tc_from_means[size_key] = tc

        tc_mean = np.mean(tc_samples)
        tc_std = np.std(tc_samples, ddof=1)
        print(f"    T_c({info['L']}) = {tc:.4f} K (Bootstrap均值曲线)")
        print(f"    Bootstrap T_c分布: {tc_mean:.4f} ± {tc_std:.4f} K")

    # 3. BKT标度拟合
    print("\n[步骤3] BKT标度律拟合...")
    size_order = ['16', '32', '64', '128', '256']
    sizes = np.array([data_info[k]['L'] for k in size_order if k in tc_from_means])
    tc_mean = np.array([tc_bootstrap_dict[k].mean() for k in size_order if k in tc_from_means])
    tc_std = np.array([tc_bootstrap_dict[k].std(ddof=1) for k in size_order if k in tc_from_means])

    T_BKT, A, T_BKT_err, x_fit, y_fit = bkt_scale_fit(tc_mean, tc_std, sizes)

    if T_BKT is not None:
        print(f"  拟合结果:")
        print(f"    T_BKT = {T_BKT:.4f} ± {T_BKT_err:.4f} K")
        print(f"    A = {A:.4f}")
        print(f"    拟合方程: T_c(L) = {T_BKT:.4f} + {A:.4f} / (ln L)^2")
    else:
        print("  警告: BKT拟合失败")
        return

    # 4. 绘制图表
    print("\n[步骤4] 绘制图表...")

    print("  绘制图2a: Bootstrap均值概率曲线(有序相和无序相)...")
    plot_figure_2a(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means)

    print("  绘制图2b: 标度变换后的概率曲线(有序相和无序相)...")
    plot_figure_2b(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means, T_BKT)

    print("  绘制图3: BKT标度律拟合...")
    plot_figure_3(tc_mean, tc_std, sizes, T_BKT, A, x_fit, y_fit)

    print("  绘制图4: 误差棒放大图...")
    plot_figure_4(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict, tc_from_means)

    print("  绘制图5: 有限尺寸效应箱线图...")
    plot_figure_5(tc_bootstrap_dict, T_BKT)

    # 5. 结果汇总
    print("\n" + "="*70)
    print("分析完成! 结果汇总:")
    print("="*70)

    print(f"\n1. 从Bootstrap均值曲线得到的T_c(L):")
    for size_key in size_order:
        if size_key in tc_from_means:
            L = data_info[size_key]['L']
            print(f"   L={L:3d}: {tc_from_means[size_key]:.4f} K")

    print(f"\n2. Bootstrap T_c(L)统计:")
    for size_key in size_order:
        if size_key in tc_bootstrap_dict:
            L = data_info[size_key]['L']
            tc_mean = np.mean(tc_bootstrap_dict[size_key])
            tc_std = np.std(tc_bootstrap_dict[size_key], ddof=1)
            print(f"   L={L:3d}: {tc_mean:.4f} ± {tc_std:.4f} K")

    print(f"\n3. BKT相变温度 (热力学极限):")
    print(f"   T_BKT = {T_BKT:.4f} ± {T_BKT_err:.4f} K")

    print(f"\n4. 标度拟合:")
    print(f"   方程: T_c(L) = {T_BKT:.4f} + {A:.4f} / (ln L)^2")

    # 保存结果
    results_file = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("BKT相变温度分析结果汇总 - 修正版V2\n")
        f.write("="*70 + "\n\n")

        f.write("1. 从Bootstrap均值曲线得到的T_c(L):\n")
        for size_key in size_order:
            if size_key in tc_from_means:
                L = data_info[size_key]['L']
                f.write(f"   L={L:3d}: {tc_from_means[size_key]:.6f} K\n")

        f.write(f"\n2. Bootstrap T_c(L)统计:\n")
        for size_key in size_order:
            if size_key in tc_bootstrap_dict:
                L = data_info[size_key]['L']
                tc_mean = np.mean(tc_bootstrap_dict[size_key])
                tc_std = np.std(tc_bootstrap_dict[size_key], ddof=1)
                f.write(f"   L={L:3d}: {tc_mean:.6f} ± {tc_std:.6f} K\n")

        f.write(f"\n3. BKT相变温度 (热力学极限):\n")
        f.write(f"   T_BKT = {T_BKT:.6f} ± {T_BKT_err:.6f} K\n")

        f.write(f"\n4. 标度拟合:\n")
        f.write(f"   方程: T_c(L) = {T_BKT:.6f} + {A:.6f} / (ln L)^2\n")

    print(f"\n详细结果已保存至: {results_file}")
    print(f"所有图表已保存至: {OUTPUT_DIR}")

    print("\n分析完成!")

if __name__ == '__main__':
    main()
