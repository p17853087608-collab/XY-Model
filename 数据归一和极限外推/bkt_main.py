"""
BKT相变温度分析 - 主流程模块
包含完整的分析流程
"""

import numpy as np
import os
from bkt_core import (
    N_BOOTSTRAP, OUTPUT_DIR, data_info, SIZE_ORDER
)
from bkt_core import (
    load_all_data, bootstrap_probability_curves, find_tc_at_50, bkt_scale_fit
)
from bkt_plotting import (
    plot_figure_2a, plot_figure_2b, plot_figure_3, plot_figure_4, plot_figure_5
)


def run_bkt_analysis():
    """
    运行完整的BKT相变温度分析流程

    返回:
        分析结果字典
    """
    print("="*70)
    print("BKT相变温度分析 - 模块化版本")
    print("="*70)

    # 1. 数据加载
    print("\n[步骤1] 加载数据...")
    temps_dict, raw_probs_ordered_dict, raw_probs_amorphous_dict = load_all_data()

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
    sizes = np.array([data_info[k]['L'] for k in SIZE_ORDER if k in tc_from_means])
    tc_mean = np.array([tc_bootstrap_dict[k].mean() for k in SIZE_ORDER if k in tc_from_means])
    tc_std = np.array([tc_bootstrap_dict[k].std(ddof=1) for k in SIZE_ORDER if k in tc_from_means])

    T_BKT, A, T_BKT_err, x_fit, y_fit = bkt_scale_fit(tc_mean, tc_std, sizes)

    if T_BKT is None:
        print("  警告: BKT拟合失败")
        return None

    print(f"  拟合结果:")
    print(f"    T_BKT = {T_BKT:.4f} ± {T_BKT_err:.4f} K")
    print(f"    A = {A:.4f}")
    print(f"    拟合方程: T_c(L) = {T_BKT:.4f} + {A:.4f} / (ln L)^2")

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
    for size_key in SIZE_ORDER:
        if size_key in tc_from_means:
            L = data_info[size_key]['L']
            print(f"   L={L:3d}: {tc_from_means[size_key]:.4f} K")

    print(f"\n2. Bootstrap T_c(L)统计:")
    for size_key in SIZE_ORDER:
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
        f.write("BKT相变温度分析结果汇总 - 模块化版本\n")
        f.write("="*70 + "\n\n")

        f.write("1. 从Bootstrap均值曲线得到的T_c(L):\n")
        for size_key in SIZE_ORDER:
            if size_key in tc_from_means:
                L = data_info[size_key]['L']
                f.write(f"   L={L:3d}: {tc_from_means[size_key]:.6f} K\n")

        f.write(f"\n2. Bootstrap T_c(L)统计:\n")
        for size_key in SIZE_ORDER:
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

    # 返回结果字典
    return {
        'T_BKT': T_BKT,
        'T_BKT_err': T_BKT_err,
        'A': A,
        'tc_from_means': tc_from_means,
        'tc_bootstrap_dict': tc_bootstrap_dict,
        'sizes': sizes,
        'tc_mean': tc_mean,
        'tc_std': tc_std,
        'x_fit': x_fit,
        'y_fit': y_fit
    }


def main():
    """主函数"""
    run_bkt_analysis()


if __name__ == '__main__':
    main()
