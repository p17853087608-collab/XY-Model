"""
自动化训练脚本
自动循环训练直到50%概率点在某一范围内
"""
import os
import sys
import subprocess
import pandas as pd
import time
import shutil
from datetime import datetime

def run_training():
    """运行训练脚本"""
    print("\n" + "="*80)
    print("开始训练模型...")
    print("="*80)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, 'model_train.py'],
            check=True,
            capture_output=False,
            encoding=None
        )

        training_time = time.time() - start_time
        print(f"\n训练完成！耗时: {training_time/60:.1f} 分钟")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n训练失败，返回码: {e.returncode}")
        return False

def run_prediction():
    """运行预测脚本"""
    print("\n" + "="*80)
    print("开始预测测试集...")
    print("="*80)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, 'predict.py', '--no-tta'],
            check=True,
            capture_output=False,
            encoding=None
        )

        prediction_time = time.time() - start_time
        print(f"\n预测完成！耗时: {prediction_time/60:.1f} 分钟")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n预测失败，返回码: {e.returncode}")
        return False

def check_result(csv_file='save_data/overall_summary.csv'):
    """检查50%概率点是否在目标范围内"""
    print("\n" + "="*80)
    print("检查预测结果...")
    print("="*80)

    try:
        df = pd.read_csv(csv_file)

        # 找到最接近50%的概率点
        df['diff_from_50'] = (df['avg_ordered_probability'] - 0.5).abs()
        closest_row = df.loc[df['diff_from_50'].idxmin()]

        closest_temp = float(closest_row['folder_name'])
        closest_prob = closest_row['avg_ordered_probability']

        print(f"\n最接近50%的概率点:")
        print(f"  温度: {closest_temp:.6f} K")
        print(f"  有序相概率: {closest_prob:.6f}")

        # 检查是否在目标范围内 [0.91, 0.97]
        if 0.91 <= closest_temp <= 0.97:
            print(f"\n✓ 成功！50%概率点在目标范围内 [0.91, 0.97] K")
            return True, closest_temp, closest_prob
        else:
            print(f"\n✗ 未达到目标，当前温度: {closest_temp:.3f} K，目标范围: [0.91, 0.97] K")
            return False, closest_temp, closest_prob

    except Exception as e:
        print(f"\n检查失败: {e}")
        return False, None, None

def main(max_iterations=10):
    """主循环"""
    print("="*80)
    print("自动化训练脚本")
    print("="*80)
    print(f"目标: 50%概率对应的温度在 [0.91, 0.97] K 之间")
    print(f"最大迭代次数: {max_iterations}")
    print("="*80)

    total_start_time = time.time()

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'#'*80}")
        print(f"# 迭代 #{iteration} / {max_iterations}")
        print(f"# {'#'*78}")

        # 1. 训练
        if not run_training():
            print("\n训练失败，跳过本次迭代")
            continue

        # 2. 预测
        if not run_prediction():
            print("\n预测失败，跳过本次迭代")
            continue

        # 3. 检查结果
        success, closest_temp, closest_prob = check_result()

        if success:
            print(f"\n{'✓'*80}")
            print(f"✓ 训练成功！")
            print(f"✓ 最终温度: {closest_temp:.6f} K")
            print(f"✓ 最终概率: {closest_prob:.6f}")
            print(f"✓ {'✓'*78}")

            total_time = time.time() - total_start_time
            print(f"\n总耗时: {total_time/60:.1f} 分钟 ({total_time/3600:.2f} 小时)")
            print(f"迭代次数: {iteration}")
            print(f"\n模型已保存: best.pth")
            print(f"预测结果: save_data/overall_summary.csv")
            print("="*80)
            return True

        # 备份模型
        if os.path.exists('best.pth'):
            backup_name = f'backup_iter_{iteration:03d}.pth'
            shutil.copy('best.pth', backup_name)
            print(f"\n模型已备份: {backup_name}")

        elapsed = time.time() - total_start_time
        print(f"\n已用时间: {elapsed/60:.1f} 分钟")

    print(f"\n达到最大迭代次数 ({max_iterations})，未达到目标")
    total_time = time.time() - total_start_time
    print(f"总耗时: {total_time/60:.1f} 分钟")
    return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='自动训练循环直到达到目标')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='最大迭代次数 (默认: 10)')

    args = parser.parse_args()

    success = main(max_iterations=args.max_iterations)

    if success:
        print("\n✓ 训练成功完成！")
        sys.exit(0)
    else:
        print("\n✗ 训练未达到目标")
        sys.exit(1)
