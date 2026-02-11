"""
BKT相变温度分析 - 核心模块
包含配置、数据加载、Bootstrap分析、BKT拟合和数据变换函数
"""

import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

# ============== 配置部分 ==============

# 数据路径配置
DATA_BASE_PATH = 'd:/LX/绘图数据'

# 定义文件夹和对应的CSV文件路径
data_info = {
    '16': {
        'path': os.path.join(DATA_BASE_PATH, '16x16/overall_summary.csv'),
        'L': 16,
        'raw_data_path': os.path.join(DATA_BASE_PATH, '16x16')
    },
    '32': {
        'path': os.path.join(DATA_BASE_PATH, '32x32/overall_summary.csv'),
        'L': 32,
        'raw_data_path': os.path.join(DATA_BASE_PATH, '32x32')
    },
    '64': {
        'path': os.path.join(DATA_BASE_PATH, '64x64/overall_summary.csv'),
        'L': 64,
        'raw_data_path': os.path.join(DATA_BASE_PATH, '64x64')
    },
    '128': {
        'path': os.path.join(DATA_BASE_PATH, '128x128/overall_summary.csv'),
        'L': 128,
        'raw_data_path': os.path.join(DATA_BASE_PATH, '128x128')
    },
    '256': {
        'path': os.path.join(DATA_BASE_PATH, '256x256/overall_summary.csv'),
        'L': 256,
        'raw_data_path': os.path.join(DATA_BASE_PATH, '256x256')
    }
}

# Bootstrap参数
N_BOOTSTRAP = 1000  # Bootstrap次数

# 输出目录配置
OUTPUT_DIR = os.path.join(DATA_BASE_PATH, 'bkt_analysis_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 绘图样式配置
MPL_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.4
}

# 颜色配置
COLORS_CONFIG = {
    'ordered': ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F'],  # 红色系（有序相）
    'amorphous': ['#1976D2', '#42A5F5', '#66BB6A', '#26A69A', '#4DB6AC'],  # 蓝色系（无序相）
    'boxplot': ['#D32F2F', '#FF6B6B', '#FF8E53', '#FFB74D', '#FFD54F'],  # 箱线图
    'other': {
        'grid': 'gray',
        'line50': '#333333',
        'bkt_line': '#D32F2F',
        'fit_line': '#E64B35',
        'marker_edge_darkred': 'darkred',
        'marker_edge_darkblue': 'darkblue'
    }
}

# 标记样式
MARKERS = ['o', 's', '^', 'D', 'p']

# 尺寸顺序
SIZE_ORDER = ['16', '32', '64', '128', '256']

# 图表文件名配置
FIGURE_FILES = {
    '2a': 'figure_2a_bootstrap_means',
    '2b': 'figure_2b_scaled_curves',
    '3': 'figure_3_bkt_fit',
    '4': 'figure_4_error_zoom',
    '5': 'figure_5_boxplot'
}


# ============== 数据加载函数 ==============

def load_summary_data(csv_path):
    """
    加载overall_summary.csv文件

    参数:
        csv_path: CSV文件路径

    返回:
        包含温度和概率数据的字典，或None（文件不存在时）
    """
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
    """
    加载每个温度点的原始概率数据(有序相和无序相)

    参数:
        folder_path: 数据文件夹路径
        temps: 温度数组

    返回:
        raw_probs_ordered: 有序相概率字典 {temp: array}
        raw_probs_amorphous: 无序相概率字典 {temp: array}
    """
    raw_probs_ordered = {}
    raw_probs_amorphous = {}
    for temp in temps:
        prob_file = os.path.join(folder_path, 'save_data', f'{temp:.3f}.csv')
        if os.path.exists(prob_file):
            df = pd.read_csv(prob_file)
            raw_probs_ordered[temp] = df['ordered_probability'].values
            raw_probs_amorphous[temp] = df['amorphous_probability'].values
    return raw_probs_ordered, raw_probs_amorphous


def load_all_data():
    """
    加载所有尺寸的数据

    返回:
        temps_dict: 温度字典 {size_key: array}
        raw_probs_ordered_dict: 有序相概率字典 {size_key: {temp: array}}
        raw_probs_amorphous_dict: 无序相概率字典 {size_key: {temp: array}}
    """
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

    return temps_dict, raw_probs_ordered_dict, raw_probs_amorphous_dict


# ============== Bootstrap分析函数 ==============

def bootstrap_probability_curves(temps, raw_probs_ordered, raw_probs_amorphous, n_bootstrap=1000):
    """
    对每个温度点执行Bootstrap重采样

    参数:
        temps: 温度数组
        raw_probs_ordered: 有序相原始概率字典
        raw_probs_amorphous: 无序相原始概率字典
        n_bootstrap: Bootstrap次数

    返回:
        prob_mean_ordered: 有序相Bootstrap均值
        prob_std_ordered: 有序相Bootstrap标准差
        prob_mean_amorphous: 无序相Bootstrap均值
        prob_std_amorphous: 无序相Bootstrap标准差
        tc_bootstrap: T_c(L)估计值数组
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
    """
    使用三次样条插值精确找到50%概率对应的温度

    参数:
        temp: 温度数组
        prob: 概率数组

    返回:
        tc: 50%概率对应的温度
        prob_at_tc: 该温度下的概率值
    """
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


def bkt_scale_fit(tc_mean, tc_std, sizes):
    """
    BKT标度律拟合: T_c(L) = T_BKT + A / (ln L)^2

    参数:
        tc_mean: T_c(L)均值数组
        tc_std: T_c(L)标准差数组
        sizes: 系统尺寸数组

    返回:
        T_BKT: BKT相变温度
        A: 拟合系数
        T_BKT_err: T_BKT误差
        x_fit: 拟合用的x值
        y_fit: 拟合用的y值
    """
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


def transform_probability_curve(temps, probs, tc_L, T_BKT):
    """
    对概率曲线进行标度变换: T' = T - [T_c(L) - T_BKT]

    参数:
        temps: 原始温度数组
        probs: 概率数组
        tc_L: 有限尺寸的相变温度
        T_BKT: BKT相变温度

    返回:
        transformed_temps: 变换后的温度数组
        probs: 概率数组（不变）
    """
    shift = tc_L - T_BKT
    transformed_temps = temps - shift
    return transformed_temps, probs
