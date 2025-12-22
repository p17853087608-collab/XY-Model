import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def integrate_and_plot():
    input_folder = "processed_csv_files_final"
    
    if not os.path.exists(input_folder):
        print(f"错误：文件夹 {input_folder} 不存在。")
        return
    
    csv_files = [f for f in os.listdir(input_folder) if f.startswith("processed_") and f.endswith(".csv")]
    print(f"找到 {len(csv_files)} 个文件: {csv_files}")
    
    all_data = []
    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"✅ 读取文件: {file}, 数据行数: {len(df)}")
        except Exception as e:
            print(f"❌ 读取失败: {file}, 错误: {e}")
    
    if not all_data:
        print("错误：没有成功读取任何文件。")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"总数据行数: {len(combined_df)}")
    print(f"L值分布: {combined_df['L'].value_counts().to_dict()}")
    
    plt.figure(figsize=(12, 8))
    L_values = sorted(combined_df['L'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    line_styles = ['-', '--', ':']
    
    for i, L in enumerate(L_values):
        subset = combined_df[combined_df['L'] == L].sort_values('T_BKT')
        
        plt.plot(subset['T_BKT'], subset['avg_ordered_probability'], 
                color=colors[i], linestyle=line_styles[i], linewidth=2,
                label=f'L={L} (有序概率)', marker='o', markersize=4, alpha=0.8)
        
        plt.plot(subset['T_BKT'], subset['avg_amorphous_probability'], 
                color=colors[i], linestyle=line_styles[i], linewidth=2,
                label=f'L={L} (非晶概率)', marker='^', markersize=4, alpha=0.8)
    
    plt.xlabel('$T_{BKT}$', fontsize=14)
    plt.ylabel('概率', fontsize=14)
    plt.title('不同L值下概率与$T_{BKT}$的关系', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc='best')
    plt.xlim(combined_df['T_BKT'].min() - 0.05, combined_df['T_BKT'].max() + 0.05)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    
    output_path = "probability_vs_tbkt_curve_final.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")
    
if __name__ == "__main__":
    integrate_and_plot()