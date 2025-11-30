from batch_spin_generator_parallel import ParallelBatchSpinGenerator

# 创建批量生成器实例（多线程+GPU加速）
batch_generator = ParallelBatchSpinGenerator(
    base_config={
        'lattice_size': 32,          # 晶格大小
        'equilibrium_steps': 2000,   # 平衡步数
        'measurement_steps': 20000,  # 测量步数
        'use_gpu': True,            # 启用GPU加速
        'random_seed': None         # 随机种子（自动生成）
    },
    max_workers=8                  # 并行线程数（建议4-8）
)

# 定义参数网格（多组参数组合）
parameter_grid = {
    'lattice_size': [32],        # 不同晶格大小
    'measurement_steps': [20000],   # 不同测量步数
    'use_gpu': [True]                    # 启用GPU加速
}

# 设置参数网格
batch_generator.define_parameter_grid(parameter_grid)


# 生成所有模拟实例配置
batch_generator.generate_all_simulations()

# 运行并行模拟（生成大量自旋图）
batch_generator.run_all_simulations_parallel(
    temperature_range=(0.1, 2.0),  # 温度范围
    num_temperatures=10            # 温度点数量
)

# 获取批量模拟汇总信息
summary = batch_generator.get_batch_summary()
print(f"总任务数: {summary['total_simulations']}")
print(f"成功数: {summary['successful_simulations']}")
print(f"生成图片总数: {summary['total_spin_images']}")
print(f"结果目录: {summary['batch_output_dir']}")