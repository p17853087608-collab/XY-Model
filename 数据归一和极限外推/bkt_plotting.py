"""
BKT相变温度分析 - 绘图模块
包含所有图表生成函数
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from bkt_core import (
    OUTPUT_DIR, MPL_CONFIG, COLORS_CONFIG, MARKERS, SIZE_ORDER, data_info, FIGURE_FILES
)

# 配置matplotlib
matplotlib.rcParams.update(MPL_CONFIG)

# 设置科学期刊风格
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')


def plot_figure_2a(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means):
    """
    Figure 2a: Bootstrap mean probability curves with error bars - ordered and amorphous phases
    MLST journal style
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    linestyles = ['-', '-', '-', '-', '-']

    # Ordered phase curves
    for idx, size_key in enumerate(SIZE_ORDER):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_ordered_dict[size_key]
        probs_std = prob_std_ordered_dict[size_key]
        tc = tc_from_means[size_key]

        ax.errorbar(temps, probs_mean, yerr=probs_std,
                   color=COLORS_CONFIG['ordered'][idx], marker=MARKERS[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle=linestyles[idx], alpha=0.85,
                   label=f'L={L} (Ordered)', fillstyle='none',
                   markeredgecolor=COLORS_CONFIG['other']['marker_edge_darkred'],
                   markeredgewidth=1.0, zorder=3)

        # Mark 50% point
        ax.plot(tc, 0.5, marker='*', markersize=18,
               color=COLORS_CONFIG['ordered'][idx], markeredgecolor='black',
               markeredgewidth=1.0, zorder=10, fillstyle='none')

    # Amorphous phase curves
    for idx, size_key in enumerate(SIZE_ORDER):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_amorphous_dict[size_key]
        probs_std = prob_std_amorphous_dict[size_key]

        ax.errorbar(temps, probs_mean, yerr=probs_std,
                   color=COLORS_CONFIG['amorphous'][idx], marker=MARKERS[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='--', alpha=0.85, dash_capstyle='round',
                   label=f'L={L} (Amorphous)', fillstyle='none',
                   markeredgecolor=COLORS_CONFIG['other']['marker_edge_darkblue'],
                   markeredgewidth=1.0, zorder=2)

    ax.axhline(y=0.5, color=COLORS_CONFIG['other']['line50'],
               linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax.set_xlabel('Temperature $T$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color=COLORS_CONFIG['other']['grid'])

    # 分开图例：有序相和无序相分成两列
    legend = ax.legend(loc='upper right', fontsize=8, ncol=2,
                     frameon=True, framealpha=0.95, fancybox=True,
                     borderpad=0.5, handlelength=2.2, columnspacing=1.2)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["2a"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["2a"]}.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_figure_2b(temps_dict, prob_mean_ordered_dict, prob_std_ordered_dict,
                   prob_mean_amorphous_dict, prob_std_amorphous_dict, tc_from_means, T_BKT):
    """
    Figure 2b: BKT scaled probability curves - ordered and amorphous phases
    MLST journal style
    """
    from bkt_core import transform_probability_curve

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Ordered phase curves
    for idx, size_key in enumerate(SIZE_ORDER):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_ordered_dict[size_key]
        probs_std = prob_std_ordered_dict[size_key]
        tc = tc_from_means[size_key]

        transformed_temps, transformed_probs = transform_probability_curve(temps, probs_mean, tc, T_BKT)

        ax.errorbar(transformed_temps, transformed_probs, yerr=probs_std,
                   color=COLORS_CONFIG['ordered'][idx], marker=MARKERS[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='-', alpha=0.85,
                   label=f'L={L} (Ordered)', fillstyle='none',
                   markeredgecolor=COLORS_CONFIG['other']['marker_edge_darkred'],
                   markeredgewidth=1.0, zorder=3)

    # Amorphous phase curves
    for idx, size_key in enumerate(SIZE_ORDER):
        if size_key not in temps_dict:
            continue

        L = data_info[size_key]['L']
        temps = temps_dict[size_key]
        probs_mean = prob_mean_amorphous_dict[size_key]
        probs_std = prob_std_amorphous_dict[size_key]
        tc = tc_from_means[size_key]

        transformed_temps, transformed_probs = transform_probability_curve(temps, probs_mean, tc, T_BKT)

        ax.errorbar(transformed_temps, transformed_probs, yerr=probs_std,
                   color=COLORS_CONFIG['amorphous'][idx], marker=MARKERS[idx],
                   markersize=6, linewidth=2.0, capsize=3,
                   linestyle='--', alpha=0.85, dash_capstyle='round',
                   label=f'L={L} (Amorphous)', fillstyle='none',
                   markeredgecolor=COLORS_CONFIG['other']['marker_edge_darkblue'],
                   markeredgewidth=1.0, zorder=2)

    ax.axhline(y=0.5, color=COLORS_CONFIG['other']['line50'],
               linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    ax.axvline(x=T_BKT, color=COLORS_CONFIG['bkt_line'],
               linestyle='--', linewidth=1.8, alpha=0.7, zorder=0)

    ax.set_xlabel('Scaled Temperature $T\'$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color=COLORS_CONFIG['other']['grid'])

    legend = ax.legend(loc='upper right', fontsize=8, ncol=2,
                     frameon=True, framealpha=0.95, fancybox=True,
                     borderpad=0.5, handlelength=2.2, columnspacing=1.2)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["2b"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["2b"]}.pdf'),
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
    ax.plot(x_fit_line, y_fit_line, color=COLORS_CONFIG['fit_line'],
            linewidth=2.5,
            label=f'$T_c(L) = {T_BKT:.4f} + {A:.4f} / (\\ln L)^2$')

    # Mark T_BKT
    ax.plot(0, T_BKT, marker='*', markersize=22,
           color=COLORS_CONFIG['fit_line'], markeredgecolor='black',
           markeredgewidth=1, zorder=10, fillstyle='none')

    ax.set_xlabel('$1/(\\ln L)^2$', fontsize=13, fontweight='bold')
    ax.set_ylabel('$T_c(L)$ (K)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8)

    legend = ax.legend(loc='best', fontsize=10, frameon=True,
                     framealpha=0.95, fancybox=True, borderpad=0.5)
    legend.get_frame().set_edgecolor('#888888')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.3)

    output_path = os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["3"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["3"]}.pdf'),
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

    # 只使用红色系（有序相）
    colors = COLORS_CONFIG['ordered']

    for idx, size_key in enumerate(SIZE_ORDER):
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
                   color=colors[idx], marker=MARKERS[idx],
                   markersize=7, linewidth=2.0, capsize=4, capthick=1.5,
                   linestyle='-', alpha=0.85, elinewidth=2.0,
                   label=f'L={L}, $T_c$={tc:.4f}K',
                   fillstyle='none', markeredgecolor=COLORS_CONFIG['other']['marker_edge_darkred'],
                   markeredgewidth=1.0, zorder=3)

        # Mark 50% point
        ax.plot(tc, 0.5, marker='*', markersize=20,
               color=colors[idx], markeredgecolor='black',
               markeredgewidth=1.0, zorder=10, fillstyle='none')

    ax.axhline(y=0.5, color=COLORS_CONFIG['other']['line50'],
               linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)

    ax.set_xlabel('Temperature $T$ (K)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Ordered Phase Probability', fontsize=13, fontweight='bold')
    ax.set_ylim(0.3, 0.7)
    ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.8, color=COLORS_CONFIG['other']['grid'])

    legend = ax.legend(loc='upper right', fontsize=9, frameon=True,
                     framealpha=0.95, fancybox=True, borderpad=0.5,
                     columnspacing=1.0)
    legend.get_frame().set_edgecolor('#333333')
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout(pad=0.4)

    output_path = os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["4"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["4"]}.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_figure_5(tc_bootstrap_dict, T_BKT):
    """
    Figure 5: Finite size effect box plot
    MLST journal style - 优化版，标注在右侧，不遮挡箱线图
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    sizes = []
    data_list = []
    means = []
    medians = []
    stds = []

    for size_key in SIZE_ORDER:
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
    meanprops = {'marker': 'D', 'markerfacecolor': COLORS_CONFIG['fit_line'],
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
    colors = COLORS_CONFIG['boxplot']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.8)
        patch.set_alpha(0.7)

    # Plot mean line
    ax.plot(range(1, len(sizes)+1), means, 'o-',
           color=COLORS_CONFIG['fit_line'], linewidth=2, markersize=7,
           markeredgecolor='#333333', markeredgewidth=1.2,
           label='Mean', zorder=5, fillstyle='none')

    # Plot T_BKT reference line
    ax.axhline(y=T_BKT, color=COLORS_CONFIG['bkt_line'], linestyle='--', linewidth=2,
              alpha=0.7, zorder=4)

    # Add standard deviation annotations - 在右侧添加，不遮挡箱线图
    for i, (L, mean, std) in enumerate(zip(sizes, means, stds)):
        x_pos = i + 1
        y_upper = mean + std
        y_lower = mean - std

        # Plot error range dotted line
        ax.plot([x_pos, x_pos], [y_lower, y_upper],
               color=COLORS_CONFIG['bkt_line'], linestyle=':', linewidth=1.5, alpha=0.6)

        # 在右侧添加标准差标注，位置根据y值错开
        annotation_x = 5.8
        annotation_y = mean
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

    output_path = os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["5"]}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{FIGURE_FILES["5"]}.pdf'),
                format='pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
