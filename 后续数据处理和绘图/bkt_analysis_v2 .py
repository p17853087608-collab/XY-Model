"""
BKT相变温度分析 - 修正版 V2
对每个温度点执行Bootstrap,计算均值和标准差
从Bootstrap均值曲线找50%概率点
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import find_peaks
import os
import sys
from io import StringIO
import warnings
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# 设置字体（根据绘图标准：Source Serif Variable）
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Source Serif Variable', 'Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 设置绘图标准（基于双栏600pt宽度）
# 双栏图：600pt ≈ 8英寸 (72pt/inch)
DPI = 600
DOUBLE_COL_WIDTH = 8.0  # 英寸
SINGLE_COL_WIDTH = 4.0  # 英寸

# 字体字号规范
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 7
matplotlib.rcParams['figure.titlesize'] = 11
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['grid.alpha'] = 0.4
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.major.width'] = 0.8
matplotlib.rcParams['ytick.major.width'] = 0.8

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
        'path': 'd:/LX/绘图数据/128x128/save_data/overall_summary.csv',
        'L': 128,
        'raw_data_path': 'd:/LX/绘图数据/128x128'
    },
    '256': {
        'path': 'd:/LX/绘图数据/256x256/save_data/overall_summary.csv',
        'L': 256,
        'raw_data_path': 'd:/LX/绘图数据/256x256'
    }
}

# Bootstrap参数
N_BOOTSTRAP = 1000  # Bootstrap次数

# 输出目录
OUTPUT_DIR = 'd:/LX/绘图数据/bkt_analysis_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 传统序参量数据目录
SUSCEPTIBILITY_DATA_DIR = 'd:/LX/绘图数据/传统序参量'

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

# ============== 传统序参量分析函数 ==============
def read_susceptibility_data(filepath):
    """读取磁化率数据文件"""
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

def cubic_spline_peak_interp(x, y):
    """使用三次样条插值找到y的峰值对应的精确x位置"""
    # 找到峰值的索引
    peak_idx = np.argmax(y)

    # 使用峰值附近的点进行三次样条插值
    left_idx = max(0, peak_idx - 5)
    right_idx = min(len(x), peak_idx + 6)
    x_sub = x[left_idx:right_idx]
    y_sub = y[left_idx:right_idx]

    # 创建三次样条插值
    cs = CubicSpline(x_sub, y_sub)

    # 找到插值函数的极值点
    x_fine = np.linspace(x_sub.min(), x_sub.max(), 10000)
    y_fine = cs(x_fine)
    peak_idx_fine = np.argmax(y_fine)
    x_peak = x_fine[peak_idx_fine]
    y_peak = y_fine[peak_idx_fine]

    # 检查插值结果是否合理
    if x_sub.min() <= x_peak <= x_sub.max():
        return x_peak, y_peak

    # 如果插值失败，返回原始值
    return x[peak_idx], y[peak_idx]

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

    # 保存所有bootstrap样本用于计算标准差
    all_bootstrap_probs_ordered = []
    all_bootstrap_probs_amorphous = []

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

        # 保存样本
        all_bootstrap_probs_ordered.append(bootstrap_probs_ordered)
        all_bootstrap_probs_amorphous.append(bootstrap_probs_amorphous)

        # 累积计算均值
        prob_mean_ordered = (prob_mean_ordered * i + bootstrap_probs_ordered) / (i + 1)
        prob_mean_amorphous = (prob_mean_amorphous * i + bootstrap_probs_amorphous) / (i + 1)

        # 找到50%概率点(使用有序相概率)
        valid_mask = ~np.isnan(bootstrap_probs_ordered)
        if np.any(valid_mask):
            tc, _ = find_tc_at_50(temps[valid_mask], bootstrap_probs_ordered[valid_mask])
            tc_bootstrap.append(tc)

        if (i + 1) % 200 == 0:
            print(f"    进度: {i+1}/{n_bootstrap}")

    # 使用NumPy直接计算标准差（修复原公式错误）
    all_bootstrap_probs_ordered = np.array(all_bootstrap_probs_ordered)
    all_bootstrap_probs_amorphous = np.array(all_bootstrap_probs_amorphous)
    prob_std_ordered = np.nanstd(all_bootstrap_probs_ordered, axis=0, ddof=1)
    prob_std_amorphous = np.nanstd(all_bootstrap_probs_amorphous, axis=0, ddof=1)

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
def bkt_scale_fit(tc_mean, tc_std, sizes, fixed_T_BKT=None):
    """BKT标度律拟合: T_c(L) = T_BKT + a / [ln(L) + b*ln(ln(L))]^2

    Args:
        tc_mean: 各尺寸的相变温度均值
        tc_std: 各尺寸的相变温度标准差
        sizes: 系统尺寸数组
        fixed_T_BKT: 如果提供,则固定T_BKT值,只拟合a和b参数
    """
    valid_mask = ~np.isnan(tc_mean) & ~np.isnan(tc_std)
    L_valid = sizes[valid_mask]
    y_valid = tc_mean[valid_mask]
    w_valid = 1.0 / (tc_std[valid_mask] ** 2)

    if len(L_valid) > 2:
        if fixed_T_BKT is not None:
            # 固定T_BKT模式:只拟合a和b
            print(f"  使用固定 T_BKT = {fixed_T_BKT:.4f}")
            T_BKT = fixed_T_BKT

            def fit_func_fixed(L, a, b):
                log_L = np.log(L)
                log_log_L = np.log(np.log(L))
                return T_BKT + a / (log_L + b * log_log_L)**2

            # 初始猜测: a在10附近, b在3.7附近
            p0 = [10.0, 3.7]
            # 参数边界: a ∈ [5, 15], b ∈ [3.5, 4.0]
            # 与自由拟合模式保持一致以确保物理意义统一
            bounds = ([5, 3.5], [15, 4.0])

            popt, pcov = curve_fit(fit_func_fixed, L_valid, y_valid, p0=p0,
                                   sigma=1/np.sqrt(w_valid), bounds=bounds,
                                   maxfev=5000)
            a, b = popt
            T_BKT_err = 0.0  # 固定参数无误差
            dof = len(L_valid) - 2  # 2个参数
        else:
            # 自由拟合模式:拟合T_BKT, a, b
            def fit_func(L, T_BKT, a, b):
                log_L = np.log(L)
                log_log_L = np.log(np.log(L))
                return T_BKT + a / (log_L + b * log_log_L)**2

            # 初始猜测: T_BKT在数据范围内, a在10附近, b在3.7附近
            p0 = [np.mean(y_valid), 10.0, 3.7]
            # 参数边界: T_BKT > 0, a ∈ [5, 15], b ∈ [3.5, 4.0]
            # 注意：a限制为正数是因为负a会使T_c(L)<T_BKT，这与物理预期相反
            # 与固定T_BKT模式保持一致以确保物理意义统一
            bounds = ([0.01, 5, 3.5], [np.inf, 15, 4.0])
            popt, pcov = curve_fit(fit_func, L_valid, y_valid, p0=p0,
                                   sigma=1/np.sqrt(w_valid), bounds=bounds,
                                   maxfev=50000)
            T_BKT, a, b = popt
            T_BKT_err = np.sqrt(np.diag(pcov))[0]
            dof = len(L_valid) - 3  # 3个参数

        # 计算拟合优度
        if fixed_T_BKT is not None:
            y_pred = fit_func_fixed(L_valid, a, b)
        else:
            y_pred = fit_func(L_valid, T_BKT, a, b)
        residuals = y_valid - y_pred
        chi2 = np.sum(w_valid * residuals**2)
        chi2_reduced = chi2 / dof if dof > 0 else np.inf

        return T_BKT, a, b, T_BKT_err, L_valid, y_valid, chi2_reduced
    else:
        return None, None, None, None, None, None, None

# ============== 数据变换函数 ==============
def transform_probability_curve(temps, probs, L, T_BKT, a, b):
    """
    对概率曲线进行BKT有限尺寸标度变换（数据塌缩）

    理论依据:
    - Hasenfratz & Niedermayer, Nucl. Phys. B 414, 785 (1994)
    - BKT有限尺寸校正公式: T_c(L) = T_BKT + a / [ln L + b ln(ln L)]^2
    - 标度变量: x = [T - T_BKT] × [ln L + b ln(ln L)]^2
    - 数据塌缩: P = f(x)，所有不同尺寸的曲线应塌缩到同一条主曲线上

    物理意义:
    - 当 T = T_c(L) 时: x = a (常数)
    - 所有不同尺寸L的曲线在 x = a 处，概率都应该 = 0.5
    - 如果数据塌缩成功，说明神经网络的输出遵循BKT标度律

    参数:
        temps: 温度数组
        probs: 概率数组
        L: 系统尺寸
        T_BKT: BKT相变温度
        a: 拟合参数a
        b: 拟合参数b

    返回:
        x: 标度变量 x = [T - T_BKT] × [ln L + b ln(ln L)]^2
        probs: 概率(保持不变)
    """
    log_L = np.log(L)
    log_log_L = np.log(np.log(L))
    x = (temps - T_BKT) * (log_L + b * log_log_L)**2
    return x, probs

# ============== 绘图函数 ==============
def plot_figure_2a(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means):
    """
    Figure 2a: Bootstrap mean probability curves with error bars - ordered and amorphous phases
    MLST journal style - 左右并排规格
    """
    # 左右并排规格: 3.3英寸宽 × 2.8英寸高
    fig, ax = plt.subplots(figsize=(3.3, 2.8), dpi=DPI)
    ax.grid(False)

    size_order = ['16', '32', '64', '128', '256']
    # 有序相用蓝色系，无序相用红色系，尺寸越大颜色越深
    colors_ordered = ['#64B5F6', '#42A5F5', '#1976D2', '#1565C0', '#0D47A1']  # 蓝色系（有序相），从浅到深
    colors_amorphous = ['#EF9A9A', '#E57373', '#F44336', '#D32F2F', '#B71C1C']  # 红色系（无序相），从浅到深
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

        ax.plot(temps, probs_mean,
               color=colors_ordered[idx], marker='o',
               markersize=3, linewidth=1.2,
               linestyle=linestyles[idx], alpha=0.85,
               label=f'L={L}', fillstyle='none',
               markeredgecolor=colors_ordered[idx], markeredgewidth=0.8,
               zorder=3)

    # Amorphous phase curves
    for idx, size_key in enumerate(size_order):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_amorphous_dict[size_key]
        probs_std = prob_std_amorphous_dict[size_key]

        ax.plot(temps, probs_mean,
               color=colors_amorphous[idx], marker='o',
               markersize=3, linewidth=1.2,
               linestyle='--', alpha=0.85, dash_capstyle='round',
               fillstyle='none',
               markeredgecolor=colors_amorphous[idx], markeredgewidth=0.8,
               zorder=2)

    ax.set_xlabel('$T$ (K)', fontsize=9)
    ax.set_ylabel('', fontsize=9)  # 去掉纵坐标标题
    ax.set_ylim(0, 1.00)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([])  # 去掉纵坐标刻度数字
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)

    # x轴范围 - 保持原来的温度范围
    ax.set_xlim(0.65, 1.45)
    ax.set_xticks([0.65, 0.85, 1.05, 1.25, 1.45])
    ax.set_xticks(np.arange(0.65, 1.451, 0.02), minor=True)
    ax.set_xticklabels([f'{t:.2f}' for t in [0.65, 0.85, 1.05, 1.25, 1.45]])
    ax.tick_params(axis='both', labelsize=9, direction='in')
    ax.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)

    # 坐标轴边框 - 与图2b一致
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    # 图例：只显示有序相，去掉边框，放在右边框左侧
    legend = ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.5), fontsize=7, ncol=1,
                     frameon=False)

    plt.tight_layout(pad=0.2, w_pad=0.1)
    
    # 设置固定的坐标轴框位置，确保与图2b一致
    # [left, bottom, width, height] (相对坐标0-1)
    ax.set_position([0.06, 0.15, 0.84, 0.75])

    output_path = os.path.join(OUTPUT_DIR, 'figure_2a_bootstrap_means.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2a_bootstrap_means.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_2b(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means, T_BKT, a, b,
                   tc_mean=None, tc_std=None, sizes=None, L_fit=None, y_fit=None, exclude_sizes=None):
    """
    Figure 2b: BKT scaled probability curves - ordered and amorphous phases
    标度变换: T' = T - [T_c(L) - T_BKT]
    MLST journal style

    Args:
        tc_mean, tc_std, sizes, L_fit, y_fit: 图3(BKT拟合)所需参数
        exclude_sizes: 要排除的尺寸列表，如 ['256']
    """
    if exclude_sizes is None:
        exclude_sizes = []

    # 左右并排规格: 3.3英寸宽 × 2.8英寸高
    fig, ax = plt.subplots(figsize=(3.3, 2.8), dpi=DPI)
    ax.grid(False)

    size_order = ['16', '32', '64', '128', '256']
    # 过滤掉要排除的尺寸
    size_order = [s for s in size_order if s not in exclude_sizes]

    # 与Figure 2a相同的配色方案
    colors_ordered = ['#64B5F6', '#42A5F5', '#1976D2', '#1565C0', '#0D47A1']  # 蓝色系（有序相），从浅到深
    colors_amorphous = ['#EF9A9A', '#E57373', '#F44336', '#D32F2F', '#B71C1C']  # 红色系（无序相），从浅到深
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

        scaled_temp, transformed_probs = transform_probability_curve(temps, probs_mean, L, T_BKT, a, b)

        ax.plot(scaled_temp, transformed_probs,
               color=colors_ordered[idx], marker='o',
               markersize=3, linewidth=1.2,
               linestyle='-', alpha=0.85,
               label=f'L={L}', fillstyle='none',
               markeredgecolor=colors_ordered[idx], markeredgewidth=0.8,
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

        scaled_temp, transformed_probs = transform_probability_curve(temps, probs_mean, L, T_BKT, a, b)

        ax.plot(scaled_temp, transformed_probs,
               color=colors_amorphous[idx], marker='o',
               markersize=3, linewidth=1.2,
               linestyle='--', alpha=0.85, dash_capstyle='round',
               fillstyle='none',
               markeredgecolor=colors_amorphous[idx], markeredgewidth=0.8,
               zorder=2)

    ax.set_xlabel("$[T - T_{BKT}] [\\ln L + b \\ln(\\ln L)]^2$", fontsize=9)
    ax.set_ylabel('', fontsize=9)
    ax.set_ylim(0, 1.00)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels([])  # 去掉主图纵坐标刻度数字
    ax.set_yticks(np.arange(0, 1.01, 0.1), minor=True)

    # 标度变量范围: 以x=a为中心,向左右各扩展30个单位,划分为4个区间
    x_min = a - 30
    x_max = a + 30
    ax.set_xlim(x_min, x_max)
    # 划分为4个区间: 左下限, 左1/3, 右1/3, 右上限,并添加x=a的刻度
    x_ticks = [x_min, x_min + (x_max - x_min)/3, a, x_max - (x_max - x_min)/3, x_max]
    ax.set_xticks(x_ticks)
    ax.set_xticks(np.arange(x_min, x_max + 1, 5), minor=True)
    # 刻度标签向下取整
    ax.set_xticklabels([f'{np.floor(t):.0f}' for t in x_ticks])
    ax.tick_params(axis='both', labelsize=9, direction='in')
    ax.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)

    # 坐标轴边框 - 与图2a一致
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    # 添加垂直线标记x=a处(P=0.5)
    ax.axvline(x=a, color='gray', linestyle=':', linewidth=1.0, alpha=0.7)

    # 插铺图：图3 (BKT拟合) 放在图2b内部右侧（原图例位置）
    if L_fit is not None and y_fit is not None and tc_std is not None:
        # 创建插铺图 [left, bottom, width, height] (相对坐标0-1)
        ax_inset = ax.inset_axes([0.73, 0.30, 0.26, 0.40])

        # 绘制数据点
        ax_inset.plot(L_fit, y_fit, 'o', markersize=3,
                     color='#1565C0',
                     fillstyle='full', markeredgecolor='#1565C0', zorder=3)

        # 拟合曲线
        def fit_func(L, T_BKT, a, b):
            log_L = np.log(L)
            log_log_L = np.log(np.log(L))
            return T_BKT + a / (log_L + b * log_log_L)**2

        L_fit_line = np.linspace(min(L_fit) * 0.6, max(L_fit) * 1.4, 200)
        y_fit_line = fit_func(L_fit_line, T_BKT, a, b)
        ax_inset.plot(L_fit_line, y_fit_line, color='#D32F2F', linewidth=1.5, zorder=2)

        # 设置插铺图样式
        ax_inset.set_xlabel('$L$', fontsize=7)
        ax_inset.set_ylabel('$T$ (K)', fontsize=7)
        ax_inset.yaxis.set_label_coords(-0.18, 0.5)  # 纵轴标签位置调整,进一步向左移动
        ax_inset.tick_params(axis='both', labelsize=7, direction='in')
        ax_inset.grid(False)

        # 坐标范围
        x_min_inset = 0
        x_max_inset = 280
        ax_inset.set_xlim(x_min_inset, x_max_inset)

        # y轴刻度 - 固定范围0.95-1.15,划分为3个区间
        y_min_inset = 0.95
        y_max_inset = 1.15
        ax_inset.set_ylim(y_min_inset, y_max_inset)

        # 纵坐标刻度 - 3个区间(4个点)
        y_ticks_inset = [y_min_inset, y_min_inset + (y_max_inset - y_min_inset)/3, y_max_inset - (y_max_inset - y_min_inset)/3, y_max_inset]
        ax_inset.set_yticks(y_ticks_inset)
        ax_inset.set_yticklabels([f'{t:.2f}' for t in y_ticks_inset])

        # 横坐标刻度 - 0-280
        x_ticks_inset = [0, 64, 128, 256]
        ax_inset.set_xticks(x_ticks_inset)
        ax_inset.set_xticklabels([f'{int(t)}' for t in x_ticks_inset])

        # 插铺图边框
        for spine in ax_inset.spines.values():
            spine.set_edgecolor('#333333')
            spine.set_linewidth(0.6)

    plt.tight_layout(pad=0.2, w_pad=0.1)
    
    # 设置固定的坐标轴框位置，确保与图2a一致
    # [left, bottom, width, height] (相对坐标0-1)
    ax.set_position([0.06, 0.15, 0.84, 0.75])

    output_path = os.path.join(OUTPUT_DIR, 'figure_2b_scaled_curves.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_2b_scaled_curves.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_3(tc_mean, tc_std, sizes, T_BKT, a, b, L_fit, y_fit):
    """
    Figure 3: BKT scaling law fitting (改进版)
    MLST journal style - 单栏图标准
    """
    fig, ax = plt.subplots(figsize=(4.17, 4.17), dpi=DPI)

    # Plot data points with error bars
    ax.errorbar(L_fit, y_fit, yerr=tc_std[:len(L_fit)],
               fmt='o', markersize=4, capsize=5,
               linewidth=1.5, color='#1565C0',
               label='Finite size $T_c(L)$',
               fillstyle='full', markeredgecolor='#1565C0')

    # Fitted curve
    L_fit_line = np.linspace(min(L_fit) * 0.6, max(L_fit) * 1.4, 200)
    def fit_func(L, T_BKT, a, b):
        log_L = np.log(L)
        log_log_L = np.log(np.log(L))
        return T_BKT + a / (log_L + b * log_log_L)**2

    y_fit_line = fit_func(L_fit_line, T_BKT, a, b)
    ax.plot(L_fit_line, y_fit_line, color='#D32F2F', linewidth=2.5,
            label='BKT fit')

    ax.set_xlabel('$L$', fontsize=9)
    ax.set_ylabel('$T$ (K)', fontsize=9)
    ax.tick_params(axis='both', labelsize=9, direction='in')
    ax.grid(False)

    # 坐标轴边框 - 与图2b一致
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    # 横坐标范围 - 0-256+延伸
    x_min = 0
    x_max = 280
    ax.set_xlim(x_min, x_max)

    # y轴刻度 - 固定范围0.95-1.15,划分为3个区间
    y_min = 0.95
    y_max = 1.15
    ax.set_ylim(y_min, y_max)

    # 添加纵坐标小刻度
    T_step = (y_max - y_min) / 20
    ax.set_yticks(np.arange(y_min, y_max + T_step/2, T_step), minor=True)
    ax.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)

    # x轴刻度 - 0-280
    x_ticks = [0, 64, 128, 256]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(t)}' for t in x_ticks])

    # y轴刻度 - 3个区间(4个点)
    y_ticks = [y_min, y_min + (y_max - y_min)/3, y_max - (y_max - y_min)/3, y_max]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{t:.2f}' for t in y_ticks])

    # 添加水平虚线标注理论临界点T_BKT
    ax.axhline(y=T_BKT, color='red', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

    # 添加图例 - 去掉T_BKT那一项
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='#1565C0', marker='o', lw=1.5, markersize=4, label='Finite size $T_c(L)$'),
        Line2D([0], [0], color='#D32F2F', lw=2.5, label='BKT fit')
    ]
    ax.legend(handles=custom_lines, fontsize=7, loc='upper right', framealpha=0.9)
    
    plt.tight_layout(pad=0.3)

    output_path = os.path.join(OUTPUT_DIR, 'figure_3_bkt_fit.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_bkt_fit.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_4(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict, tc_from_means):
    """
    Figure 4: Zoom-in of error bars near 50% probability point
    Bootstrap standard deviation distribution (ordered phase only)
    MLST journal style - 单栏图标准
    """
    fig, ax = plt.subplots(figsize=(4.17, 4.5), dpi=DPI)
    ax.grid(False)

    size_order = ['16', '32', '64', '128', '256']
    # 使用蓝色系（有序相），尺寸越大颜色越深
    colors = ['#64B5F6', '#42A5F5', '#1976D2', '#1565C0', '#0D47A1']
    markers = ['o', 'o', 'o', 'o', 'o']  # 统一使用小圆点

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
        ax.plot(temps_zoom, probs_mean_zoom,
               color=colors[idx], marker=markers[idx],
               markersize=7, linewidth=2.0,
               linestyle='-', alpha=0.85,
               label=f'$T_c$={tc:.3f}K',
               fillstyle='none', markeredgecolor=colors[idx],
               markeredgewidth=1.0, zorder=3)

        # 在P=0.5处加菱形标记（灰黑色系）
        ax.scatter(tc, 0.5, marker='D', s=80, color='#455A64',
                  edgecolor='#263238', linewidth=1.5, zorder=5)

    ax.axhline(y=0.5, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax.set_xlabel('$T$ (K)', fontsize=9)
    ax.set_ylabel('Probability', fontsize=9)
    ax.tick_params(axis='both', labelsize=9, direction='in')
    ax.set_ylim(0.3, 0.7)

    # 横坐标划分为4个区间
    all_temps = []
    for size_key in size_order:
        if size_key in temps_dict:
            tc = tc_from_means[size_key]
            all_temps.extend([tc - 0.1, tc + 0.1])
    x_min = min(all_temps)
    x_max = max(all_temps)
    ax.set_xlim(x_min, x_max)

    # x轴刻度: 4个区间 (左下限, 左1/3, 右1/3, 右上限)
    x_ticks = [x_min, x_min + (x_max - x_min)/3, x_max - (x_max - x_min)/3, x_max]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{t:.2f}' for t in x_ticks])

    # 启用x轴小刻度,在大刻度之间均匀分布
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))  # 每个主区间分为4等分
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)

    # 设置纵坐标主刻度和标签
    y_major_ticks = [0.30, 0.40, 0.50, 0.60, 0.70]
    ax.set_yticks(y_major_ticks)
    ax.set_yticklabels([f'{t:.2f}' for t in y_major_ticks])

    # 启用y轴小刻度,在大刻度之间均匀分布
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))  # 每个主区间分为4等分
    ax.tick_params(axis='y', which='minor', length=2, width=0.5, direction='in', labelleft=False)

    legend = ax.legend(loc='center right', bbox_to_anchor=(0.98, 0.8), fontsize=7, frameon=False)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, 'figure_4_error_zoom.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_error_zoom.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

def plot_figure_5(tc_bootstrap_dict, T_BKT, figure_6_axes_pos=None):
    """
    Figure 5: Bootstrap T_c(L) vs Susceptibility Peak T_peak(L) comparison
    MLST journal style - 箱线图 + 峰值折线对比
    """
    size_order = ['16', '32', '64', '128', '256']
    sizes = []
    data_list = []
    bootstrap_means = []
    bootstrap_stds = []

    # 收集Bootstrap数据
    for size_key in size_order:
        if size_key not in tc_bootstrap_dict:
            continue
        L = data_info[size_key]['L']
        sizes.append(L)
        data_list.append(tc_bootstrap_dict[size_key])
        bootstrap_means.append(np.mean(tc_bootstrap_dict[size_key]))
        bootstrap_stds.append(np.std(tc_bootstrap_dict[size_key], ddof=1))

    # 打印箱线图误差分析
    print("\n" + "="*70)
    print("Figure 5 箱线图误差分析 (Bootstrap T_c分布):")
    print("="*70)
    print(f"{'L':<6} {'均值(K)':<12} {'标准差(K)':<12} {'变异系数(%)':<12} {'Q1':<10} {'Q3':<10} {'IQR':<10}")
    print("-"*70)

    for idx, size_key in enumerate(size_order):
        if size_key not in tc_bootstrap_dict:
            continue
        L = sizes[idx]
        bootstrap_data = tc_bootstrap_dict[size_key]
        mean_val = bootstrap_means[idx]
        std_val = bootstrap_stds[idx]
        cv = 100 * std_val / mean_val  # 变异系数
        q1 = np.percentile(bootstrap_data, 25)
        q3 = np.percentile(bootstrap_data, 75)
        iqr = q3 - q1

        print(f"{L:<6} {mean_val:<12.6f} {std_val:<12.6f} {cv:<12.4f} {q1:<10.6f} {q3:<10.6f} {iqr:<10.6f}")

    # 分析误差变化趋势
    print("\n误差变化趋势分析:")
    for i in range(len(bootstrap_stds)-1):
        L1, L2 = sizes[i], sizes[i+1]
        std1, std2 = bootstrap_stds[i], bootstrap_stds[i+1]
        std_change_pct = 100 * (std2 - std1) / std1
        print(f"  L={L1} -> L={L2}: 标准差变化 {std_change_pct:+.2f}%")

    # 读取磁化率峰值数据
    peak_temps = []

    data_dict = {}
    for L in sizes:
        filepath = os.path.join(SUSCEPTIBILITY_DATA_DIR, f'{L}_susceptibility_results.txt')
        if os.path.exists(filepath):
            temps, chi, chi_std = read_susceptibility_data(filepath)
            data_dict[L] = {'temps': temps, 'chi': chi, 'chi_std': chi_std}

    for idx, L in enumerate(sizes):
        if L in data_dict:
            temps = data_dict[L]['temps']
            chi = data_dict[L]['chi']

            # 使用三次样条插值找到峰值
            peak_temp, _ = cubic_spline_peak_interp(temps, chi)
            peak_temps.append(peak_temp)

    # 绘图 - 左右并排规格: 3.3英寸宽 × 2.8英寸高
    fig, ax = plt.subplots(figsize=(3.3, 2.8), dpi=DPI)

    # 绘制Bootstrap箱线图
    boxprops = {'linewidth': 0.8, 'color': '#0D47A1'}
    whiskerprops = {'linewidth': 0.8, 'color': '#1976D2'}
    capprops = {'linewidth': 0.8, 'color': '#1976D2'}
    medianprops = {'linewidth': 1.2, 'color': '#0D47A1'}
    meanprops = {'marker': 'o', 'markerfacecolor': '#E64B35',
                'markeredgecolor': '#333333', 'markersize': 0, 'markeredgewidth': 0.8}

    box_plot = ax.boxplot(data_list, labels=[f'{L}' for L in sizes],
                        positions=sizes,
                        patch_artist=True, showmeans=True, showfliers=False,
                        boxprops=boxprops,
                        whiskerprops=whiskerprops,
                        capprops=capprops,
                        medianprops=medianprops,
                        meanprops=meanprops,
                        widths=10.0)

    # 统一浅蓝色配色（无渐变）
    uniform_color = '#64B5F6'
    for patch in box_plot['boxes']:
        patch.set_facecolor(uniform_color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(0.8)
        patch.set_alpha(0.6)

    # 绘制Bootstrap均值折线（连接箱体）
    ax.plot(sizes, bootstrap_means, 'o-',
           color='#0D47A1', linewidth=1.2, markersize=4,
           markeredgecolor='#333333', markeredgewidth=0.8,
           zorder=5, label='Bootstrap $T_c(L)$')

    # 绘制磁化率峰值折线（无误差棒）
    ax.plot(sizes[:len(peak_temps)], peak_temps,
            's--', color='#D32F2F', linewidth=1.2, markersize=4,
            label='$\\chi$ Peak $T_{peak}(L)$',
            zorder=6)

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#0D47A1', linewidth=1.2, label='Bootstrap $T_c(L)$'),
        Line2D([0], [0], color='#D32F2F', marker='s', linestyle='-', linewidth=1.2,
                markersize=4, label='$\\chi$ Peak $T_{peak}(L)$')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=7, frameon=False)

    ax.set_xlabel('$L$', fontsize=9)
    ax.set_ylabel('$T$ (K)', fontsize=9)
    ax.tick_params(axis='both', labelsize=9, direction='in')
    ax.grid(False)

    # 调整Y轴范围
    all_temps = bootstrap_means + peak_temps
    y_min = min(all_temps) - 0.01
    y_max = max(all_temps) + 0.01
    ax.set_ylim(y_min, y_max)

    # y轴刻度: 5个区间 (下限, 1/4, 1/2, 3/4, 上限)
    y_ticks = [y_min, y_min + (y_max - y_min)/4, y_min + (y_max - y_min)/2,
               y_max - (y_max - y_min)/4, y_max]
    ax.set_yticks(y_ticks)
    ax.set_yticks(np.arange(y_min, y_max + 0.001, (y_max - y_min)/20), minor=True)
    ax.set_yticklabels([f'{t:.2f}' for t in y_ticks])

    # x轴刻度: 使用与图6相同的间隔
    x_ticks = sizes
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{int(L)}' for L in sizes])

    # 添加纵坐标小刻度
    ax.tick_params(axis='both', which='minor', length=2, width=0.5, direction='in', labelleft=False, labelbottom=False)

    # 坐标轴边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    # 使用图6的坐标轴框位置，确保坐标轴框宽高一致
    if figure_6_axes_pos is not None:
        ax.set_position(figure_6_axes_pos)
        print(f"Using Figure 6 axes position for Figure 5")
    else:
        plt.tight_layout(pad=0.2)

    output_path = os.path.join(OUTPUT_DIR, 'figure_5_boxplot.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_5_boxplot.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.close()

def plot_magnetization_comparison(tc_bootstrap_dict, figure_5_axes_pos=None):
    """
    绘制图6: 不同尺寸L下的磁化率曲线对比
    """
    print("\n=== 生成图6: 磁化率曲线对比 ===")
    """
    Figure 6: Susceptibility vs Temperature with Bootstrap prediction markers
    使用脚本运行时得到的Bootstrap数据
    """
    sizes = [16, 32, 64, 128, 256]
    size_order = ['16', '32', '64', '128', '256']

    # 读取磁化率数据
    data_dict = {}
    for L in sizes:
        filepath = os.path.join(SUSCEPTIBILITY_DATA_DIR, f'{L}_susceptibility_results.txt')
        if os.path.exists(filepath):
            temps, chi, chi_std = read_susceptibility_data(filepath)
            data_dict[L] = {'temps': temps, 'chi': chi, 'chi_std': chi_std}

    # 使用脚本运行时得到的Bootstrap T_c(L)
    T_cnn_dict = {}
    for size_key in size_order:
        if size_key in tc_bootstrap_dict:
            L = data_info[size_key]['L']
            T_cnn_dict[L] = np.mean(tc_bootstrap_dict[size_key])

    # 找到磁化率峰值
    peak_temps = []
    peak_chis = []

    print("\nSusceptibility Peaks (使用三次样条插值):")
    for L in sizes:
        if L not in data_dict:
            continue
        temps = data_dict[L]['temps']
        chi = data_dict[L]['chi']

        # 使用三次样条插值找到精确的峰值
        peak_temp, peak_chi_val = cubic_spline_peak_interp(temps, chi)

        peak_temps.append(peak_temp)
        peak_chis.append(peak_chi_val)

        if L in T_cnn_dict:
            t_cnn = T_cnn_dict[L]
            diff = t_cnn - peak_temp
            rel_diff = 100 * diff / peak_temp
            print(f"  L={L:3d}: T_peak = {peak_temp:.6f} K, Bootstrap T_c = {t_cnn:.6f} K, ΔT = {diff:+.6f} K ({rel_diff:+.3f}%)")

    # 绘图: 磁化率曲线 - 左右并排规格: 3.3英寸宽 × 2.8英寸高
    fig, ax = plt.subplots(figsize=(3.3, 2.8), dpi=DPI)

    colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#1976D2', '#0D47A1']

    for idx, L in enumerate(sizes):
        if L not in data_dict or L not in T_cnn_dict:
            continue

        data = data_dict[L]
        temps = data['temps']
        chi = data['chi']
        t_cnn = T_cnn_dict[L]

        # 画磁化率曲线
        ax.plot(temps, chi, color=colors[idx], linewidth=1.2, linestyle='-', zorder=2, alpha=0.9)

        # 画数据点（每隔几个点画一个）
        step = max(1, len(temps) // 20)
        ax.scatter(temps[::step], chi[::step], color=colors[idx], marker='o', s=12, alpha=0.5, zorder=3)

        # 标记峰值位置
        peak_idx = sizes.index(L)
        peak_temp = peak_temps[peak_idx]
        chi_at_peak = np.interp(peak_temp, temps, chi)
        ax.plot(peak_temp, chi_at_peak, marker='*', markersize=10, color=colors[idx],
                markeredgecolor='black', markeredgewidth=1.0, zorder=6)

        # 标记Bootstrap预测的T_c(L)位置
        if min(temps) <= t_cnn <= max(temps):
            chi_at_cnn = np.interp(t_cnn, temps, chi)
            ax.plot(t_cnn, chi_at_cnn, marker='D', markersize=6, color=colors[idx],
                    markeredgecolor='black', markeredgewidth=1.0, fillstyle='none', zorder=5)

    ax.set_xlabel('$T$ (K)', fontsize=9)
    ax.set_ylabel('', fontsize=9)  # 隐藏默认的ylabel
    ax.tick_params(axis='both', labelsize=9, which='major', direction='in')
    ax.grid(False)

    # 使用对数坐标
    ax.set_yscale('log')
    all_temps = np.concatenate([data_dict[L]['temps'] for L in data_dict])
    all_chi = np.concatenate([data_dict[L]['chi'] for L in data_dict])
    ax.set_xlim(0.65, 1.45)
    # x轴刻度: 固定范围0.65-1.45，划分为4个区间
    x_min, x_max = 0.65, 1.45
    x_ticks = [x_min, x_min + (x_max - x_min)/4, x_min + (x_max - x_min)/2,
               x_max - (x_max - x_min)/4, x_max]
    ax.set_xticks(x_ticks)
    ax.set_xticks(np.arange(x_min, x_max + 0.01, 0.04), minor=True)
    # 显示所有刻度
    ax.set_xticklabels([f'{t:.2f}' for t in x_ticks])

    # 添加x轴小刻度
    ax.tick_params(axis='x', which='minor', length=2, width=0.5, direction='in', labelbottom=False)

    # 坐标轴边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.6)

    min_chi = min([data['chi'][data['chi'] > 0].min() for data in data_dict.values()])
    ax.set_ylim(min_chi * 0.5, max(all_chi) * 1.5)

    # 添加图例 - 原有图例 + 图2a的图例(下方靠右)
    from matplotlib.lines import Line2D
    # L=64的颜色是colors[2] = '#42A5F5'
    star_marker = Line2D([0], [0], marker='*', markersize=10, color='#42A5F5',
                         markeredgecolor='black', markeredgewidth=1.0, linestyle='None')
    diamond_marker = Line2D([0], [0], marker='D', markersize=6, color='#42A5F5',
                            markeredgecolor='black', markeredgewidth=1.0, fillstyle='none', linestyle='None')

    # 菱形图例 - 位置不变
    legend_diamond = ax.legend([diamond_marker], ['Bootstrap $T_c(L)$'],
                               loc='upper right', bbox_to_anchor=(0.99, 1.0),
                               fontsize=7, ncol=1, frameon=False)
    # 添加菱形图例到坐标轴中
    ax.add_artist(legend_diamond)

    # 五角星图例 - 位于L图例上方,x坐标0.99
    legend_star = ax.legend([star_marker], ['$\\chi_{peak}$'],
                            loc='center right', bbox_to_anchor=(0.99, 0.85),
                            fontsize=7, ncol=1, frameon=False)
    # 添加五角星图例到坐标轴中
    ax.add_artist(legend_star)

    # 第二行图例: 图2a的图例内容(L=16, 32, 64, 128, 256)
    colors_ordered = ['#64B5F6', '#42A5F5', '#1976D2', '#1565C0', '#0D47A1']
    legend_lines = []
    for i, L in enumerate([16, 32, 64, 128, 256]):
        line = Line2D([0], [0], color=colors_ordered[i], marker='o', markersize=3,
                      linestyle='-', linewidth=1.2, fillstyle='none',
                      markeredgecolor=colors_ordered[i], markeredgewidth=0.8, label=f'L={L}')
        legend_lines.append(line)

    # 使用bbox_to_anchor精确控制位置: 放在右侧,在X_peak图例下方
    # x坐标设置为1.0,y坐标调到0.65
    legend_lower = ax.legend(legend_lines, [f'L={L}' for L in [16, 32, 64, 128, 256]],
                             loc='center right', bbox_to_anchor=(1.0, 0.65),
                             fontsize=7, ncol=1, frameon=False)

    # 添加第三个图例到坐标轴中
    ax.add_artist(legend_lower)

    # 设置纵坐标标签
    ax.set_ylabel('$\\chi$', fontsize=9, labelpad=10)

    output_path = os.path.join(OUTPUT_DIR, 'figure_6_magnetization_comparison.pdf')
    plt.savefig(output_path, format='pdf', bbox_inches='tight', bbox_extra_artists=[legend_diamond, legend_star, legend_lower], facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'figure_6_magnetization_comparison.tiff'),
                format='tiff', dpi=600, bbox_inches='tight', bbox_extra_artists=[legend_diamond, legend_star, legend_lower], facecolor='white')
    print(f"Saved: {output_path}")

    # 获取并保存图6的坐标轴框位置
    figure_6_axes_pos = ax.get_position()
    print(f"Figure 6 axes position: {figure_6_axes_pos}")
    plt.close()

    return figure_6_axes_pos

# ============== 数据保存函数 ==============
def save_all_data(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict,
                   tc_bootstrap_dict, tc_from_means, T_BKT, a, b, L_fit, y_fit,
                   chi2_reduced, timestamp_str):
    """
    保存所有得到的数据到文件
    包括: 原始数据、中间结果、拟合参数等
    """
    # 创建数据子目录
    data_save_dir = os.path.join(OUTPUT_DIR, 'saved_data')
    os.makedirs(data_save_dir, exist_ok=True)

    print(f"\n[数据保存] 正在保存所有数据到 {data_save_dir}...")

    # 1. 保存完整数据字典 (pickle格式 - 保留所有Python对象)
    all_data = {
        'timestamp': timestamp_str,
        'temps_dict': temps_dict,
        'prob_mean_ordered_dict': prob_mean_ordered_dict,
        'prob_std_ordered_dict': prob_std_ordered_dict,
        'prob_mean_amorphous_dict': prob_mean_amorphous_dict,
        'prob_std_amorphous_dict': prob_std_amorphous_dict,
        'tc_bootstrap_dict': tc_bootstrap_dict,
        'tc_from_means': tc_from_means,
        'T_BKT': T_BKT,
        'a': a,
        'b': b,
        'L_fit': L_fit,
        'y_fit': y_fit,
        'chi2_reduced': chi2_reduced
    }

    pickle_file = os.path.join(data_save_dir, f'all_data_{timestamp_str}.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  已保存完整数据: {pickle_file}")

    # 2. 保存为CSV格式 (方便查看和进一步分析)
    # 合并所有尺寸的概率数据
    combined_data = []
    for size_key, temps in temps_dict.items():
        L = data_info[size_key]['L']
        for i, temp in enumerate(temps):
            combined_data.append({
                'L': L,
                'T': temp,
                'ordered_mean': prob_mean_ordered_dict[size_key][i],
                'ordered_std': prob_std_ordered_dict[size_key][i],
                'amorphous_mean': prob_mean_amorphous_dict[size_key][i],
                'amorphous_std': prob_std_amorphous_dict[size_key][i]
            })

    combined_df = pd.DataFrame(combined_data)
    csv_file = os.path.join(data_save_dir, f'combined_probabilities_{timestamp_str}.csv')
    combined_df.to_csv(csv_file, index=False, float_format='%.6f')
    print(f"  已保存合并概率数据: {csv_file}")

    # 3. 保存T_c数据
    tc_data = []
    size_order = ['16', '32', '64', '128', '256']
    for size_key in size_order:
        if size_key in tc_bootstrap_dict:
            L = data_info[size_key]['L']
            tc_mean_val = np.mean(tc_bootstrap_dict[size_key])
            tc_std_val = np.std(tc_bootstrap_dict[size_key], ddof=1)
            tc_from_mean = tc_from_means[size_key]
            tc_data.append({
                'L': L,
                'Tc_from_mean_curve': tc_from_mean,
                'Tc_bootstrap_mean': tc_mean_val,
                'Tc_bootstrap_std': tc_std_val,
                'Tc_bootstrap_median': np.median(tc_bootstrap_dict[size_key]),
                'Tc_bootstrap_q25': np.percentile(tc_bootstrap_dict[size_key], 25),
                'Tc_bootstrap_q75': np.percentile(tc_bootstrap_dict[size_key], 75)
            })

    tc_df = pd.DataFrame(tc_data)
    tc_csv_file = os.path.join(data_save_dir, f'Tc_data_{timestamp_str}.csv')
    tc_df.to_csv(tc_csv_file, index=False, float_format='%.6f')
    print(f"  已保存T_c数据: {tc_csv_file}")

    # 4. 保存BKT拟合参数
    fit_params = {
        'T_BKT': float(T_BKT),
        'a': float(a),
        'b': float(b),
        'chi2_reduced': float(chi2_reduced),
        'equation': f'T_c(L) = {T_BKT:.6f} + {a:.6f} / [ln L + {b:.6f}*ln(ln L)]^2'
    }

    json_file = os.path.join(data_save_dir, f'BKT_fit_params_{timestamp_str}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(fit_params, f, indent=2, ensure_ascii=False)
    print(f"  已保存BKT拟合参数: {json_file}")

    # 5. 保存Bootstrap采样数据 (每个尺寸的所有1000个T_c值)
    bootstrap_samples_file = os.path.join(data_save_dir, f'bootstrap_Tc_samples_{timestamp_str}.csv')
    bootstrap_samples_data = {}
    for size_key in size_order:
        if size_key in tc_bootstrap_dict:
            L = data_info[size_key]['L']
            bootstrap_samples_data[f'L_{L}'] = tc_bootstrap_dict[size_key]

    # 转换为DataFrame (不同尺寸可能有略微不同的样本数)
    max_length = max(len(samples) for samples in bootstrap_samples_data.values())
    for key in bootstrap_samples_data:
        samples = bootstrap_samples_data[key]
        if len(samples) < max_length:
            bootstrap_samples_data[key] = np.pad(samples, (0, max_length - len(samples)), constant_values=np.nan)

    bootstrap_df = pd.DataFrame(bootstrap_samples_data)
    bootstrap_df.to_csv(bootstrap_samples_file, index=False, float_format='%.6f')
    print(f"  已保存Bootstrap采样数据: {bootstrap_samples_file}")

    # 6. 保存拟合曲线数据 (用于重新绘图)
    def fit_func(L, T_BKT, a, b):
        log_L = np.log(L)
        log_log_L = np.log(np.log(L))
        return T_BKT + a / (log_L + b * log_log_L)**2

    L_fit_line = np.linspace(min(L_fit) * 0.6, max(L_fit) * 1.4, 200)
    y_fit_line = fit_func(L_fit_line, T_BKT, a, b)

    fit_curve_data = pd.DataFrame({
        'L': L_fit_line,
        'T_fitted': y_fit_line
    })
    fit_curve_file = os.path.join(data_save_dir, f'BKT_fit_curve_{timestamp_str}.csv')
    fit_curve_data.to_csv(fit_curve_file, index=False, float_format='%.6f')
    print(f"  已保存BKT拟合曲线数据: {fit_curve_file}")

    # 7. 保存运行配置信息
    config_info = {
        'timestamp': timestamp_str,
        'N_BOOTSTRAP': N_BOOTSTRAP,
        'sizes_analyzed': [data_info[k]['L'] for k in size_order if k in temps_dict],
        'data_paths': {k: v['path'] for k, v in data_info.items()},
        'fixed_T_BKT_value': 0.8929,
        'script_version': 'V2_with_data_saving'
    }

    config_file = os.path.join(data_save_dir, f'config_{timestamp_str}.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"  已保存运行配置: {config_file}")

    print(f"\n✓ 所有数据已成功保存到: {data_save_dir}")
    print(f"  使用的数据保存时间戳: {timestamp_str}")

    return data_save_dir

# ============== 主流程 ==============
def main():
    # 重定向stdout以捕获所有打印信息
    print_capture = StringIO()
    original_stdout = sys.stdout
    sys.stdout = print_capture

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

    # 3. BKT标度拟合 - 固定T_BKT模式(使用全部5个数据点)
    print("\n[步骤3] BKT标度律拟合(固定T_BKT模式, 全部5个数据点)...")
    size_order_all = ['16', '32', '64', '128', '256']
    sizes = np.array([data_info[k]['L'] for k in size_order_all if k in tc_from_means])
    tc_mean = np.array([tc_bootstrap_dict[k].mean() for k in size_order_all if k in tc_from_means])
    tc_std = np.array([tc_bootstrap_dict[k].std(ddof=1) for k in size_order_all if k in tc_from_means])

    # 固定T_BKT模式: 只拟合a和b两个参数
    # 对于经典二维XY模型, 文献值T_BKT ≈ 0.8929
    fixed_T_BKT_value = 0.8929
    T_BKT, a, b, T_BKT_err, L_fit, y_fit, chi2_reduced = bkt_scale_fit(
        tc_mean, tc_std, sizes, fixed_T_BKT=fixed_T_BKT_value)

    if T_BKT is not None:
        print(f"  拟合结果(固定T_BKT模式, 全部5个数据点):")
        print(f"    T_BKT = {T_BKT:.4f} K (固定值)")
        print(f"    a = {a:.4f}")
        print(f"    b = {b:.4f}")
        print(f"    chi2/df = {chi2_reduced:.4f}")
        print(f"    拟合方程: T_c(L) = {T_BKT:.4f} + {a:.4f} / [ln L + {b:.4f}*ln(ln L)]^2")
        print(f"    拟合数据点: L = {sizes}")
    else:
        print("  警告: BKT拟合失败")
        return

    # 4. 绘制图表
    print("\n[步骤4] 绘制图表...")

    print("  绘制图2a: Bootstrap均值概率曲线(有序相和无序相)...")
    plot_figure_2a(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means)

    print("  绘制图2b: BKT有限尺寸标度变换后的概率曲线(全部5个尺寸,含图3插铺图)...")
    plot_figure_2b(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means, T_BKT, a, b,
                   tc_mean=tc_mean, tc_std=tc_std, sizes=sizes, L_fit=L_fit, y_fit=y_fit,
                   exclude_sizes=None)  # 绘制所有尺寸,包含图3插铺图

    print("  绘制图3: BKT标度律拟合(自由拟合模式)...")
    plot_figure_3(tc_mean, tc_std, sizes, T_BKT, a, b, L_fit, y_fit)

    print("  绘制图4: 误差棒放大图...")
    plot_figure_4(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict, tc_from_means)

    print("  绘制图6: 磁化率曲线对比图...")
    figure_6_axes_pos = plot_magnetization_comparison(tc_bootstrap_dict, None)

    print("  绘制图5: Bootstrap T_c(L) vs Susceptibility Peak对比图...")
    plot_figure_5(tc_bootstrap_dict, T_BKT, figure_6_axes_pos)

    # 5. 结果汇总
    print("\n" + "="*70)
    print("分析完成! 结果汇总:")
    print("="*70)

    # 定义完整的尺寸顺序(用于结果汇总)
    size_order_all = ['16', '32', '64', '128', '256']

    print("\n1. 从Bootstrap均值曲线得到的T_c(L):")
    for size_key in size_order_all:
        if size_key in tc_from_means:
            L = data_info[size_key]['L']
            print(f"   L={L:3d}: {tc_from_means[size_key]:.4f} K")

    print(f"\n2. Bootstrap T_c(L)统计:")
    for size_key in size_order_all:
        if size_key in tc_bootstrap_dict:
            L = data_info[size_key]['L']
            tc_mean_val = np.mean(tc_bootstrap_dict[size_key])
            tc_std_val = np.std(tc_bootstrap_dict[size_key], ddof=1)
            print(f"   L={L:3d}: {tc_mean_val:.4f} ± {tc_std_val:.4f} K")

    print(f"\n3. BKT相变温度 (热力学极限):")
    print(f"   T_BKT = {T_BKT:.4f} ± {T_BKT_err:.4f} K")

    print(f"\n4. 标度拟合:")
    print(f"   方程: T_c(L) = {T_BKT:.4f} + {a:.4f} / [ln L + {b:.4f}*ln(ln L)]^2")
    print(f"   chi2/df = {chi2_reduced:.4f}")

    # 6. 保存所有数据
    print("\n[步骤5] 保存所有计算数据...")
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    data_save_dir = save_all_data(
        temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
        prob_mean_amorphous_dict, prob_std_amorphous_dict,
        tc_bootstrap_dict, tc_from_means, T_BKT, a, b, L_fit, y_fit,
        chi2_reduced, timestamp_str
    )

    # 保存结果到结果汇总文件
    results_file = os.path.join(OUTPUT_DIR, 'results_summary.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("BKT相变温度分析结果汇总 - 修正版V2\n")
        f.write("="*70 + "\n\n")
        f.write(f"运行时间: {timestamp_str}\n")
        f.write(f"数据保存目录: {data_save_dir}\n\n")

        f.write("1. 从Bootstrap均值曲线得到的T_c(L):\n")
        for size_key in size_order_all:
            if size_key in tc_from_means:
                L = data_info[size_key]['L']
                f.write(f"   L={L:3d}: {tc_from_means[size_key]:.6f} K\n")

        f.write(f"\n2. Bootstrap T_c(L)统计:\n")
        for size_key in size_order_all:
            if size_key in tc_bootstrap_dict:
                L = data_info[size_key]['L']
                tc_mean = np.mean(tc_bootstrap_dict[size_key])
                tc_std = np.std(tc_bootstrap_dict[size_key], ddof=1)
                f.write(f"   L={L:3d}: {tc_mean:.6f} ± {tc_std:.6f} K\n")

        f.write(f"\n3. BKT相变温度 (热力学极限):\n")
        f.write(f"   T_BKT = {T_BKT:.6f} K (固定值)\n")

        f.write(f"\n4. 标度拟合:\n")
        f.write(f"   方程: T_c(L) = {T_BKT:.6f} + {a:.6f} / [ln L + {b:.6f}*ln(ln L)]^2\n")
        f.write(f"   chi2/df = {chi2_reduced:.6f}\n")

        f.write(f"\n5. 保存的文件:\n")
        f.write(f"   - 完整数据(pickle): all_data_{timestamp_str}.pkl\n")
        f.write(f"   - 合并概率数据(CSV): combined_probabilities_{timestamp_str}.csv\n")
        f.write(f"   - T_c数据(CSV): Tc_data_{timestamp_str}.csv\n")
        f.write(f"   - BKT拟合参数(JSON): BKT_fit_params_{timestamp_str}.json\n")
        f.write(f"   - Bootstrap采样(CSV): bootstrap_Tc_samples_{timestamp_str}.csv\n")
        f.write(f"   - 拟合曲线(CSV): BKT_fit_curve_{timestamp_str}.csv\n")
        f.write(f"   - 运行配置(JSON): config_{timestamp_str}.json\n")

    print(f"\n详细结果已保存至: {results_file}")
    print(f"所有图表已保存至: {OUTPUT_DIR}")

    print("\n分析完成!")

    # 恢复stdout并保存打印信息到Markdown文件
    sys.stdout = original_stdout
    print_output = print_capture.getvalue()

    # 保存为Markdown文档
    md_file = os.path.join(OUTPUT_DIR, 'run_output.md')
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# BKT相变温度分析 - 运行输出\n\n")
        f.write("```\n")
        f.write(print_output)
        f.write("```\n")

    print(f"\n打印信息已保存至: {md_file}")

if __name__ == '__main__':
    main()
