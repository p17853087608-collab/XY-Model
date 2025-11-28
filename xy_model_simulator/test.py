from xy_model_simulator import XYModelSimulator

# 1. 创建模拟器实例
simulator = XYModelSimulator(
    lattice_size=16,           # 16x16晶格
    equilibrium_steps=1000,    # 平衡步数
    measurement_steps=5000,    # 测量步数
    random_seed=42             # 随机种子，确保可重现性
)

# 2. 运行模拟（自动生成所有结果到带时间戳的文件夹）
results = simulator.run_simulation(
    temperature_range=(0.1, 2.5),  # 温度范围
    num_temperatures=10            # 温度点数
)

# 3. 获取结果文件夹路径
print(f"模拟完成，结果保存在: {simulator.get_output_directory()}")