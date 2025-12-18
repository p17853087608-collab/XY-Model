import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('overall_summary.csv')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
plt.figure(figsize=(15, 8))

# 绘制两条折线
plt.plot(df['folder_name'], df['avg_ordered_probability'], 'b-o', label='Ordered phase (有序相)', linewidth=2, markersize=4)
plt.plot(df['folder_name'], df['avg_amorphous_probability'], 'r-s', label='Amorphous phase (非晶相)', linewidth=2, markersize=4)

# 设置图表标题和标签
plt.title('不同文件夹下的平均预测概率变化', fontsize=16, fontweight='bold')
plt.xlabel('文件夹名称', fontsize=12)
plt.ylabel('平均概率', fontsize=12)

# 设置坐标轴范围
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例
plt.legend(fontsize=11)

# 旋转x轴标签以避免重叠
plt.xticks(rotation=45, ha='right')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('probability_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('probability_analysis.pdf', format='pdf', bbox_inches='tight')

print("图表已保存为: probability_analysis.png 和 probability_analysis.pdf")
plt.show()

# 打印关键转折点信息
print("\n关键分析结果:")
print("=" * 50)

# 找到概率交叉点（如果有的话）
diff = df['avg_ordered_probability'] - df['avg_amorphous_probability']
crossing_points = df[diff.abs() < 0.1]  # 差异小于0.1的点

if not crossing_points.empty:
    print("概率接近的转折点:")
    for _, row in crossing_points.iterrows():
        print(f"  文件夹 {row['folder_name']}: 有序相={row['avg_ordered_probability']:.3f}, 非晶相={row['avg_amorphous_probability']:.3f}")

# 显示极值点
print(f"\n有序相概率最高: 文件夹 {df.loc[df['avg_ordered_probability'].idxmax(), 'folder_name']} (概率={df['avg_ordered_probability'].max():.3f})")
print(f"非晶相概率最高: 文件夹 {df.loc[df['avg_amorphous_probability'].idxmax(), 'folder_name']} (概率={df['avg_amorphous_probability'].max():.3f})")

# 分析总体趋势
print(f"\n总体趋势分析:")
print(f"  前半段 (0.800-1.050) 平均有序相概率: {df[df['folder_name'] <= 1.050]['avg_ordered_probability'].mean():.3f}")
print(f"  前半段 (0.800-1.050) 平均非晶相概率: {df[df['folder_name'] <= 1.050]['avg_amorphous_probability'].mean():.3f}")
print(f"  后半段 (1.050-1.300) 平均有序相概率: {df[df['folder_name'] > 1.050]['avg_ordered_probability'].mean():.3f}")
print(f"  后半段 (1.050-1.300) 平均非晶相概率: {df[df['folder_name'] > 1.050]['avg_amorphous_probability'].mean():.3f}")