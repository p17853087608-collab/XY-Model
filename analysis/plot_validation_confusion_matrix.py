# -*- coding: utf-8 -*-
"""
验证集混淆矩阵可视化脚本
从各尺度的验证集数据生成混淆矩阵图
数据来源：
- 16x16, 64x64, 128x128, 256x256: 数据保存/detailed_training_data_clean.csv
- 32x32: validation_confusion_matrix/validation_predictions.csv (模型重新预测)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置字体（与plot_confusion_matrix.py保持一致）
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Source Serif Variable', 'Times New Roman', 'DejaVu Serif']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 字体字号规范
matplotlib.rcParams['font.size'] = 9
matplotlib.rcParams['axes.labelsize'] = 9
matplotlib.rcParams['axes.titlesize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9
matplotlib.rcParams['legend.fontsize'] = 7

# ==================== 配置参数 ====================
BASE_DIR = r"d:\LX\绘图数据"
SCALES = ['16x16', '32x32', '64x64', '128x128', '256x256']
OUTPUT_DIR = os.path.join(BASE_DIR, "validation_confusion_matrix_plots")

# 图形尺寸设置
FIG_WIDTH = 3.3
FIG_HEIGHT = 2.8
DPI = 600

# 类别映射
CLASS_NAMES = ['Quasi-ordered\nPhase', 'Disordered\nPhase']
CLASS_NAMES_CN = ['有序相', '无序相']


def load_validation_data_16_64_128_256(scale_dir):
    """
    从detailed_training_data_clean.csv加载验证集数据
    适用于: 16x16, 64x64, 128x128, 256x256
    只提取最后一个epoch的数据，避免重复计数
    """
    csv_path = os.path.join(scale_dir, "数据保存", "detailed_training_data_clean.csv")
    
    if not os.path.exists(csv_path):
        print(f"  文件不存在: {csv_path}")
        return None, None
    
    df = pd.read_csv(csv_path, low_memory=False)
    
    # 筛选验证集数据
    val_df = df[df['phase'] == 'val'].copy()
    
    if len(val_df) == 0:
        print(f"  验证集数据为空")
        return None, None
    
    # 只提取最后一个epoch的数据
    max_epoch = val_df['epoch'].max()
    last_epoch_df = val_df[val_df['epoch'] == max_epoch]
    
    print(f"  总epoch数: {max_epoch}, 使用第 {max_epoch} epoch的数据")
    
    y_true = last_epoch_df['actual'].values
    y_pred = last_epoch_df['predicted'].values
    
    return y_true, y_pred


def load_validation_data_32x32(scale_dir):
    """
    从validation_predictions.csv加载32x32验证集数据
    """
    csv_path = os.path.join(scale_dir, "validation_confusion_matrix", "validation_predictions.csv")
    
    if not os.path.exists(csv_path):
        print(f"  文件不存在: {csv_path}")
        return None, None
    
    df = pd.read_csv(csv_path)
    
    y_true = df['true_label'].values
    y_pred = df['predicted_label'].values
    
    return y_true, y_pred


def load_validation_data(scale):
    """
    根据尺度选择合适的数据加载方法
    """
    scale_dir = os.path.join(BASE_DIR, scale)
    
    if scale == '32x32':
        return load_validation_data_32x32(scale_dir)
    else:
        return load_validation_data_16_64_128_256(scale_dir)


def compute_metrics(y_true, y_pred):
    """
    计算分类指标
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def plot_confusion_matrix(cm, scale_name, output_path, use_chinese=False):
    """
    绘制混淆矩阵热力图
    """
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    
    # 绘制热力图
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                yticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                ax=ax, cbar_kws={'shrink': 0.8})
    
    # 手动添加数值标注
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=7, color=text_color)
    
    # 设置标签
    if use_chinese:
        ax.set_xlabel('预测类别', fontsize=9)
        ax.set_ylabel('真实类别', fontsize=9)
    else:
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
    
    # 设置标题
    title = f'{scale_name} Validation Set'
    ax.set_title(title, fontsize=9, pad=8)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.tiff', '.pdf'), dpi=DPI, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  已保存: {os.path.basename(output_path)}")


def plot_combined_confusion_matrices(cms, scale_names, output_path, use_chinese=False):
    """
    将5个混淆矩阵拼接成一张图
    - 第一行: 16x16, 32x32, 64x64 (无colorbar)
    - 第二行: 128x128, 256x256 (仅256x256有colorbar)
    """
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(9.7, 6), dpi=DPI)
    gs = GridSpec(2, 4, figure=fig, wspace=0.35, hspace=0.35,
                  width_ratios=[1, 1, 0.05, 1])
    
    # 找到最大值
    vmax = max(cm.max() for cm in cms)
    
    # 位置: 
    first_row = [(0, 0), (0, 1), (0, 3)]
    second_row = [(1, 0), (1, 1)]
    
    # 绘制第一行3个图(无colorbar): 16x16, 32x32, 64x64
    for idx, pos in enumerate(first_row):
        ax = fig.add_subplot(gs[pos])
        cm = cms[idx]
        scale_name = scale_names[idx]
        
        cm_percent = cm.astype('float') / cm.sum() * 100
        
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                    yticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                    ax=ax, cbar=False, vmin=0, vmax=vmax)
        
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                percent = cm_percent[i, j]
                text = f'{count}\n({percent:.1f}%)'
                text_color = 'white' if cm[i, j] > vmax * 0.5 else 'black'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center', fontsize=7, color=text_color)
        
        ax.set_xlabel('Predicted Label', fontsize=9)
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_title(scale_name, fontsize=9, pad=8)
        ax.set_aspect('equal')
    
    # 绘制第二行: 128x128(无colorbar)
    ax = fig.add_subplot(gs[second_row[0]])
    cm = cms[3]
    scale_name = scale_names[3]
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                yticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                ax=ax, cbar=False, vmin=0, vmax=vmax)
    
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            text_color = 'white' if cm[i, j] > vmax * 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center', fontsize=7, color=text_color)
    
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)
    ax.set_title(scale_name, fontsize=9, pad=8)
    ax.set_aspect('equal')
    
    # 256x256(带colorbar)
    ax = fig.add_subplot(gs[second_row[1]])
    cm = cms[4]
    scale_name = scale_names[4]
    cm_percent = cm.astype('float') / cm.sum() * 100
    
    im = sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                yticklabels=CLASS_NAMES_CN if use_chinese else CLASS_NAMES,
                ax=ax, cbar=False, vmin=0, vmax=vmax)
    
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            percent = cm_percent[i, j]
            text = f'{count}\n({percent:.1f}%)'
            text_color = 'white' if cm[i, j] > vmax * 0.5 else 'black'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center', fontsize=7, color=text_color)
    
    ax.set_xlabel('Predicted Label', fontsize=9)
    ax.set_ylabel('True Label', fontsize=9)
    ax.set_title(scale_name, fontsize=9, pad=8)
    ax.set_aspect('equal')
    
    # 手动添加colorbar紧贴256x256右侧
    cbar_ax = fig.add_subplot(gs[1, 2])
    cbar = fig.colorbar(im.collections[0], cax=cbar_ax)
    cbar.set_label('Count', fontsize=9)
    
    # 保存
    if os.path.exists(output_path):
        os.remove(output_path)
    pdf_path = output_path.replace('.tiff', '.pdf')
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(pdf_path, dpi=DPI, 
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"  已保存拼接图: {os.path.basename(output_path)}")


def main():
    """主函数"""
    print("=" * 60)
    print("验证集混淆矩阵可视化 - 5个尺度")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n输出目录: {OUTPUT_DIR}")
    
    # 存储所有尺度的指标和数据
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_cms = []
    
    # 处理每个尺度
    for scale in SCALES:
        print(f"\n{'='*40}")
        print(f"处理尺度: {scale}")
        print(f"{'='*40}")
        
        # 加载验证集数据
        y_true, y_pred = load_validation_data(scale)
        
        if y_true is None:
            print(f"  跳过: 无法加载 {scale} 的验证集数据")
            continue
        
        print(f"  验证集样本数量: {len(y_true)}")
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        print(f"  混淆矩阵:\n{cm}")
        
        # 保存混淆矩阵用于拼接图
        all_cms.append(cm)
        
        # 收集所有数据用于合并
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # 计算指标
        metrics = compute_metrics(y_true, y_pred)
        metrics['scale'] = scale
        all_metrics.append(metrics)
        
        print(f"  准确率: {metrics['accuracy']:.4f}")
        print(f"  精确率: {metrics['precision']:.4f}")
        print(f"  召回率: {metrics['recall']:.4f}")
        print(f"  F1分数: {metrics['f1']:.4f}")
        
        # 绘制单个混淆矩阵
        cm_path = os.path.join(OUTPUT_DIR, f"{scale}_validation_confusion_matrix.tiff")
        plot_confusion_matrix(cm, scale, cm_path, use_chinese=False)
    
    # 生成5个尺度的拼接混淆矩阵图
    if len(all_cms) == 5:
        print(f"\n{'='*60}")
        print("生成5尺度拼接混淆矩阵图")
        print(f"{'='*60}")
        combined_all_path = os.path.join(OUTPUT_DIR, "validation_confusion_matrix_all_scales.tiff")
        plot_combined_confusion_matrices(all_cms, SCALES, combined_all_path, use_chinese=False)
    
    # 生成合并的混淆矩阵
    if len(all_y_true) > 0:
        print(f"\n{'='*60}")
        print("生成合并混淆矩阵（五合一）")
        print(f"{'='*60}")
        
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        print(f"  总验证集样本数量: {len(all_y_true)}")
        
        # 计算合并后的混淆矩阵
        cm_combined = confusion_matrix(all_y_true, all_y_pred)
        print(f"  合并混淆矩阵:\n{cm_combined}")
        
        # 计算合并后的指标
        metrics_combined = compute_metrics(all_y_true, all_y_pred)
        print(f"  准确率: {metrics_combined['accuracy']:.4f}")
        print(f"  精确率: {metrics_combined['precision']:.4f}")
        print(f"  召回率: {metrics_combined['recall']:.4f}")
        print(f"  F1分数: {metrics_combined['f1']:.4f}")
        
        # 绘制合并的混淆矩阵
        combined_path = os.path.join(OUTPUT_DIR, "validation_confusion_matrix_combined.tiff")
        plot_confusion_matrix(cm_combined, "Combined Validation", combined_path, use_chinese=False)
    
    # 保存汇总表格
    print(f"\n{'='*60}")
    print("保存汇总数据")
    print(f"{'='*60}")
    
    summary_df = pd.DataFrame(all_metrics)
    if len(summary_df) > 0:
        summary_df = summary_df[['scale', 'accuracy', 'precision', 'recall', 'f1', 
                                 'specificity', 'tp', 'tn', 'fp', 'fn']]
        summary_df.columns = ['尺度', '准确率', '精确率', '召回率', 'F1分数',
                              '特异度', 'TP', 'TN', 'FP', 'FN']
        
        # 添加合并后的指标
        if len(all_y_true) > 0:
            combined_row = pd.DataFrame([{
                '尺度': 'Combined',
                '准确率': metrics_combined['accuracy'],
                '精确率': metrics_combined['precision'],
                '召回率': metrics_combined['recall'],
                'F1分数': metrics_combined['f1'],
                '特异度': metrics_combined['specificity'],
                'TP': metrics_combined['tp'],
                'TN': metrics_combined['tn'],
                'FP': metrics_combined['fp'],
                'FN': metrics_combined['fn']
            }])
            summary_df = pd.concat([summary_df, combined_row], ignore_index=True)
        
        summary_path = os.path.join(OUTPUT_DIR, "validation_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"  已保存: validation_metrics_summary.csv")
        
        # 打印汇总表格
        print("\n各尺度验证集性能汇总:")
        print(summary_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
