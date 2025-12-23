from xy_model_simulator_parallel import ParallelXYModelSimulator

# ===== 在这里修改参数 =====
# 模拟参数设置
LATTICE_SIZE = 16              # 晶格大小 16x16, 32x32, 64x64 等
EQUILIBRIUM_STEPS = 1000       # 平衡步数
MEASUREMENT_STEPS = 5000       # 测量步数  
INTERACTION_CONSTANT = 1.0     # 相互作用常数J
RANDOM_SEED = 42               # 随机种子 (None表示随机)
NUM_PROCESSES = 4               # 并行进程数 (1-8)

# 温度设置 - 三选一
TEMPERATURE_MODE = "list"      # "range", "list", 或 "critical"

# 方式1: 温度范围
TEMP_MIN = 0.1                  # 最小温度
TEMP_MAX = 2.5                  # 最大温度  
NUM_TEMPERATURES = 20          # 温度点数

# 方式2: 自定义温度点列表 (当TEMPERATURE_MODE="list"时使用)
CUSTOM_TEMPERATURES = [0.1, 0.3, 0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5]

# 方式3: 临界区域精细扫描 (当TEMPERATURE_MODE="critical"时使用)
TC_CENTER = 1.1                 # 临界温度中心
TC_RANGE = 0.6                  # 扫描范围
TC_RESOLUTION = 0.02            # 温度分辨率
# ========================

# 创建模拟器
simulator = ParallelXYModelSimulator(
    lattice_size=LATTICE_SIZE,
    equilibrium_steps=EQUILIBRIUM_STEPS,
    measurement_steps=MEASUREMENT_STEPS,
    interaction_constant=INTERACTION_CONSTANT,
    random_seed=RANDOM_SEED,
    num_processes=NUM_PROCESSES
)

print(f"XY模型模拟设置:")
print(f"  晶格大小: {LATTICE_SIZE}x{LATTICE_SIZE}")
print(f"  平衡步数: {EQUILIBRIUM_STEPS}")
print(f"  测量步数: {MEASUREMENT_STEPS}")
print(f"  相互作用常数: {INTERACTION_CONSTANT}")
print(f"  随机种子: {RANDOM_SEED}")
print(f"  并行进程数: {NUM_PROCESSES}")
print(f"  温度模式: {TEMPERATURE_MODE}")

# 根据选择的温度模式运行模拟
if TEMPERATURE_MODE == "range":
    print(f"  温度范围: {TEMP_MIN} - {TEMP_MAX}, 共{NUM_TEMPERATURES}个点")
    results = simulator.run_simulation(
        temperature_range=(TEMP_MIN, TEMP_MAX),
        num_temperatures=NUM_TEMPERATURES
    )
    
elif TEMPERATURE_MODE == "list":
    print(f"  自定义温度点: {len(CUSTOM_TEMPERATURES)}个")
    print(f"  温度值: {[f'{t:.2f}' for t in CUSTOM_TEMPERATURES]}")
    
    # 对于自定义温度点，需要稍微修改
    # 这里我们用一个近似的方式
    temp_min, temp_max = min(CUSTOM_TEMPERATURES), max(CUSTOM_TEMPERATURES)
    num_temps = len(CUSTOM_TEMPERATURES)
    
    results = simulator.run_simulation(
        temperature_range=(temp_min, temp_max),
        num_temperatures=num_temps
    )
    
elif TEMPERATURE_MODE == "critical":
    # 生成临界区域的温度点
    critical_temps = np.arange(TC_CENTER - TC_RANGE/2, TC_CENTER + TC_RANGE/2, TC_RESOLUTION)
    print(f"  临界区域扫描: {TC_CENTER - TC_RANGE/2:.2f} - {TC_CENTER + TC_RANGE/2:.2f}")
    print(f"  温度分辨率: {TC_RESOLUTION}, 共{len(critical_temps)}个点")
    
    temp_min, temp_max = critical_temps[0], critical_temps[-1]
    num_temps = len(critical_temps)
    
    results = simulator.run_simulation(
        temperature_range=(temp_min, temp_max),
        num_temperatures=num_temps
    )

print(f"\n模拟完成！")
print(f"结果保存在: {simulator.base_output_dir}")

# 显示一些基本信息
if results:
    temps = results['temperature']
    energies = results['energy']
    mags = results['magnetization']
    
    print(f"\n结果摘要:")
    print(f"  温度范围: {temps[0]:.3f} - {temps[-1]:.3f}")
    print(f"  能量范围: {energies.min():.6f} - {energies.max():.6f}")
    print(f"  磁化强度范围: {mags.min():.6f} - {mags.max():.6f}")
    
    # 找到磁化率和比热的峰值位置
    sus_peak_idx = np.argmax(results['susceptibility'])
    heat_peak_idx = np.argmax(results['specific_heat'])
    
    print(f"  磁化率峰值: {results['susceptibility'][sus_peak_idx]:.6f} @ T={temps[sus_peak_idx]:.3f}")
    print(f"  比热峰值: {results['specific_heat'][heat_peak_idx]:.6f} @ T={temps[heat_peak_idx]:.3f}")
    
    if 'timing' in results:
        timing = results['timing']
        print(f"\n性能统计:")
        print(f"  总耗时: {timing.get('total_time', 'N/A'):.2f} 秒")
        if 'parallel_efficiency' in timing:
            print(f"  并行效率: {timing['parallel_efficiency']:.1f}%")