from xy_model_simulator import XYModelSimulator

# 1. 创建模拟器实例
simulator = XYModelSimulator(
    lattice_size=16,           # 16x16晶格
    equilibrium_steps=1000,    # 平衡步数
    measurement_steps=5000,    # 测量步数
    random_seed=42             # 随机种子，确保可重现性
)

# 2. 运行模拟
results = simulator.run_simulation(
    temperature_range=(0.1, 2.5),  # 温度范围
    num_temperatures=20            # 温度点数量
)

# 3. 生成结果
simulator.plot_results(output_dir='results')          # 绘制物理量图表
simulator.generate_spin_visualization(output_dir='spin_plots')  # 生成自旋配置图
simulator.save_results(filename='simulation_data.txt', output_dir='results')  # 保存数据