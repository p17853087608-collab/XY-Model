from xy_model_simulator_optimized import XYModelSimulator
simulator = XYModelSimulator(
    lattice_size=16,
    equilibrium_steps=2000,
    measurement_steps=20000,
    interaction_constant=1.0,
    random_seed=None,
    use_gpu=False,
)

# 相变区域精细扫描
results = simulator.run_simulation(
    temperature_range=(0.1, 0.8),
    num_temperatures=500
)

print(f"模拟完成，结果保存在: {simulator.get_output_directory()}")