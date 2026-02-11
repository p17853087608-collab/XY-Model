#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XY模型并行模拟调参脚本 - 简化版本
支持实时进度显示、详细数据保存和分析报告生成
"""

import os
import time
from xy_model_simulator_parallel import ParallelXYModelSimulator

def main():
    print("=" * 60)
    print("XY模型并行模拟器 - 调参运行脚本")
    print("=" * 60)
    
    # ==================== 在这里修改参数 ====================
    
    # 温度设置 - 三种模式选择一种：
    TEMPERATURE_MODE = "range"  # "range", "custom", "critical"
    
    # 模式1: 温度范围
    TEMPERATURE_RANGE = (0.1,2.0)  # (最低温度, 最高温度)
    NUM_TEMPERATURES = 20             # 温度点数量
    
    # 模式2: 自定义温度点
    CUSTOM_TEMPERATURES = [
        0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 
        1.25, 1.30, 1.35, 1.40, 1.45
    ]
    
    # 模式3: 临界区域密集采样
    CRITICAL_CENTER = 0.89  # 估计临界温度
    CRITICAL_RANGE = 0.3     # 临界区域范围
    
    # 模拟参数
    LATTICE_SIZE = 16                # 晶格大小
    EQUILIBRIUM_STEPS = 2000         # 平衡步数
    MEASUREMENT_STEPS = 100          # 测量步数
    INTERACTION_CONSTANT = 1.0       # 相互作用常数
    
    # 并行计算参数
    NUM_PROCESSES = 20                 # 并行进程数 (建议 <= CPU核心数)
    # 生成高度随机的种子：结合当前时间、进程ID和随机数
    # 注意：os已在文件顶部导入，这里只导入需要的其他模块
    import random
    import datetime
    # 确保每次运行都有完全不同的随机种子，即使同时运行多个脚本
    base_time = datetime.datetime.now().timestamp() * 1000000
    process_id = os.getpid() % 100000
    random_offset = random.randint(0, 999999)
    RANDOM_SEED = int((base_time + process_id + random_offset) % 1000000)
    
    # 输出设置
    SAVE_RESULTS = True               # 是否保存结果
    VERBOSE = True                    # 是否显示详细信息
    
    # ==================== 参数设置结束 ====================
    
    print(f"当前配置:")
    print(f"  晶格大小: {LATTICE_SIZE}×{LATTICE_SIZE}")
    print(f"  平衡步数: {EQUILIBRIUM_STEPS}")
    print(f"  测量步数: {MEASUREMENT_STEPS}")
    print(f"  并行进程数: {NUM_PROCESSES}")
    print(f"  温度模式: {TEMPERATURE_MODE}")
    
    # 设置温度
    if TEMPERATURE_MODE == "range":
        temperatures = None  # 使用范围
        temp_desc = f"范围 {TEMPERATURE_RANGE[0]:.2f}-{TEMPERATURE_RANGE[1]:.2f}, {NUM_TEMPERATURES}个点"
    elif TEMPERATURE_MODE == "custom":
        temperatures = CUSTOM_TEMPERATURES
        temp_desc = f"自定义列表: {len(temperatures)}个点"
    elif TEMPERATURE_MODE == "critical":
        # 在临界区域密集采样
        fine_range = 0.1
        coarse_range = (CRITICAL_RANGE - fine_range) / 2
        temps = []
        # 低温区稀疏采样
        for t in [CRITICAL_CENTER - CRITICAL_RANGE + i*0.05 
                  for i in range(int(coarse_range/0.05))]:
            if t >= CRITICAL_CENTER - CRITICAL_RANGE:
                temps.append(t)
        # 临界区密集采样
        for t in [CRITICAL_CENTER - fine_range + i*0.02 
                  for i in range(int(2*fine_range/0.02))]:
            temps.append(t)
        # 高温区稀疏采样
        for t in [CRITICAL_CENTER + fine_range + i*0.05 
                  for i in range(int(coarse_range/0.05))]:
            if t <= CRITICAL_CENTER + CRITICAL_RANGE:
                temps.append(t)
        temperatures = sorted(list(set(temps)))
        temp_desc = f"临界区域密集采样: {len(temperatures)}个点"
    else:
        raise ValueError(f"未知的温度模式: {TEMPERATURE_MODE}")
    
    print(f"  温度设置: {temp_desc}")
    print()
    
    # 创建模拟器实例
    print("初始化并行模拟器...")
    simulator = ParallelXYModelSimulator(
        lattice_size=LATTICE_SIZE,
        equilibrium_steps=EQUILIBRIUM_STEPS,
        measurement_steps=MEASUREMENT_STEPS,
        interaction_constant=INTERACTION_CONSTANT,
        random_seed=RANDOM_SEED,
        num_processes=NUM_PROCESSES
    )
    
    # 开始模拟
    print("开始并行模拟...")
    print("-" * 60)
    start_time = time.time()
    
    try:
        # 运行模拟
        if temperatures is None:
            # 使用温度范围
            results = simulator.run_parallel_simulation(
                temperature_range=TEMPERATURE_RANGE,
                num_temperatures=NUM_TEMPERATURES
            )
        else:
            # 使用自定义温度列表
            results = simulator.run_parallel_simulation(
                temperature_list=temperatures
            )
        
        end_time = time.time()
        
        print("-" * 60)
        print("模拟完成！")
        
        # 显示结果摘要
        print(f"\n结果摘要:")
        temps = results['temperature']
        energy = results['energy']
        mag = results['magnetization']
        sus = results['susceptibility']
        heat = results['specific_heat']
        
        print(f"  温度范围: {temps.min():.3f} - {temps.max():.3f}")
        print(f"  能量范围: {energy.min():.6f} - {energy.max():.6f}")
        print(f"  磁化强度范围: {mag.min():.6f} - {mag.max():.6f}")
        print(f"  磁化率峰值: {sus.max():.6f} @ T={temps[sus.argmax()]:.3f}")
        print(f"  比热峰值: {heat.max():.6f} @ T={temps[heat.argmax()]:.3f}")
        
        # 临界温度估计
        tc_sus = temps[sus.argmax()]
        tc_heat = temps[heat.argmax()]
        tc_avg = (tc_sus + tc_heat) / 2
        print(f"  临界温度估计: {tc_avg:.3f}")
        
        # 性能统计
        timing = results.get('timing', {})
        if timing:
            print(f"\n性能统计:")
            print(f"  总耗时: {end_time - start_time:.2f} 秒")
            print(f"  记录耗时: {timing.get('total_time', 0):.2f} 秒")
            print(f"  平均每温度点: {timing.get('avg_time_per_temp', 0):.2f} 秒")
            print(f"  并行效率: {timing.get('parallel_efficiency', 0):.1f}%")
        
        # 保存位置
        if hasattr(simulator, 'base_output_dir'):
            print(f"\n结果已保存到:")
            print(f"  {simulator.base_output_dir}")

            # 列出主要文件
            if os.path.exists(simulator.base_output_dir):
                print(f"\n保存的文件:")
                for file in sorted(os.listdir(simulator.base_output_dir)):
                    file_path = os.path.join(simulator.base_output_dir, file)
                    if os.path.isfile(file_path):
                        size_mb = os.path.getsize(file_path) / (1024*1024)
                        print(f"  - {file} ({size_mb:.2f} MB)")
        
    except KeyboardInterrupt:
        print("\n\n模拟被用户中断")
        return
    except Exception as e:
        print(f"\n\n模拟过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()