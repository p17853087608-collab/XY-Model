"""
XY模型模拟器测试脚本
测试进度显示功能、GPU加速、结果文件管理等所有功能
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径（如果脚本不在项目根目录）
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from xy_model_simulator import XYModelSimulator


def test_basic_simulation():
    """测试基础模拟功能"""
    print("=" * 60)
    print("测试1: 基础模拟功能")
    print("=" * 60)
    
    # 创建模拟器
    simulator = XYModelSimulator(
        lattice_size=8,            # 小晶格，快速测试
        equilibrium_steps=500,     # 减少步数
        measurement_steps=1000,     # 减少步数
        random_seed=42
    )
    
    # 运行模拟
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.0), 
        num_temperatures=5
    )
    
    # 验证结果
    assert 'temperature' in results
    assert 'energy' in results
    assert 'magnetization' in results
    assert 'timing' in results
    
    print("✓ 基础模拟功能测试通过")
    print(f"结果保存在: {simulator.get_output_directory()}")
    return simulator


def test_gpu_acceleration():
    """测试GPU加速功能（如果可用）"""
    print("=" * 60)
    print("测试2: GPU加速功能")
    print("=" * 60)
    
    try:
        # 测试GPU版本
        simulator_gpu = XYModelSimulator(
            lattice_size=16,
            equilibrium_steps=1000,
            measurement_steps=5000,
            use_gpu=True,
            random_seed=123
        )
        
        start_time = time.time()
        results_gpu = simulator_gpu.run_simulation(
            temperature_range=(0.1, 2.0), 
            num_temperatures=8
        )
        gpu_time = time.time() - start_time
        
        # 测试CPU版本
        simulator_cpu = XYModelSimulator(
            lattice_size=16,
            equilibrium_steps=1000,
            measurement_steps=5000,
            use_gpu=False,
            random_seed=123
        )
        
        start_time = time.time()
        results_cpu = simulator_cpu.run_simulation(
            temperature_range=(0.1, 2.0), 
            num_temperatures=8
        )
        cpu_time = time.time() - start_time
        
        # 验证结果一致性
        assert np.allclose(results_gpu['temperature'], results_cpu['temperature'])
        assert np.allclose(results_gpu['energy'], results_cpu['energy'], rtol=1e-3)
        
        # 比较性能
        speedup = cpu_time / gpu_time
        print(f"GPU加速测试通过")
        print(f"CPU时间: {cpu_time:.2f}秒")
        print(f"GPU时间: {gpu_time:.2f}秒")
        print(f"加速比: {speedup:.2f}x")
        
        return simulator_gpu, simulator_cpu
        
    except ImportError:
        print("⚠ CuPy未安装，跳过GPU加速测试")
        return None, None


def test_progress_display():
    """测试进度显示功能"""
    print("=" * 60)
    print("测试3: 进度显示功能")
    print("=" * 60)
    
    # 创建模拟器
    simulator = XYModelSimulator(
        lattice_size=8,
        equilibrium_steps=500,
        measurement_steps=1000,
        random_seed=456
    )
    
    # 运行模拟（会显示进度）
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.5), 
        num_temperatures=10
    )
    
    # 验证进度数据
    assert 'timing' in results
    timing = results['timing']
    
    assert 'total_time' in timing
    assert 'per_temperature_time' in timing
    assert 'min_time' in timing
    assert 'max_time' in timing
    
    print("✓ 进度显示功能测试通过")
    print(f"总耗时: {timing['total_time']:.2f}秒")
    print(f"平均耗时: {timing['avg_time_per_temp']:.2f}秒")
    print(f"最快: {timing['min_time']:.2f}秒")
    print(f"最慢: {timing['max_time']:.2f}秒")
    
    return simulator


def test_multiple_runs():
    """测试多次运行自动区分功能"""
    print("=" * 60)
    print("测试4: 多次运行自动区分")
    print("=" * 60)
    
    results = []
    directories = []
    
    # 运行3次模拟
    for i in range(3):
        print(f"第{i+1}次运行...")
        
        simulator = XYModelSimulator(
            lattice_size=8,
            equilibrium_steps=500,
            measurement_steps=1000,
            random_seed=789 + i  # 不同的随机种子
        )
        
        result = simulator.run_simulation(
            temperature_range=(0.1, 1.5), 
            num_temperatures=5
        )
        
        results.append(result)
        directories.append(simulator.get_output_directory())
        
        # 等待一小段时间确保时间戳不同
        time.sleep(0.1)
    
    # 验证所有结果文件夹不同
    assert len(set(directories)) == 3, "多次运行结果文件夹未正确区分"
    
    print("✓ 多次运行自动区分测试通过")
    for i, dir_path in enumerate(directories):
        print(f"第{i+1}次结果: {dir_path}")
    
    return results, directories


def test_spin_visualization():
    """测试自旋可视化功能"""
    print("=" * 60)
    print("测试5: 自旋可视化功能")
    print("=" * 60)
    
    # 创建模拟器
    simulator = XYModelSimulator(
        lattice_size=8,
        equilibrium_steps=500,
        measurement_steps=1000,
        random_seed=999
    )
    
    # 运行模拟
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.0), 
        num_temperatures=5
    )
    
    # 验证自旋配置数据
    assert hasattr(simulator, 'spin_configurations')
    assert len(simulator.spin_configurations) > 0
    
    # 检查自旋图片文件是否存在
    spin_dir = os.path.join(simulator.get_output_directory(), 'spin_configurations')
    spin_files = [f for f in os.listdir(spin_dir) if f.endswith('.png')]
    assert len(spin_files) == 5, f"自旋图片数量不正确: {len(spin_files)}"
    
    print("✓ 自旋可视化功能测试通过")
    print(f"生成了 {len(spin_files)} 张自旋图片")
    for spin_file in spin_files[:3]:  # 只显示前3个
        print(f"  - {spin_file}")
    
    return simulator


def test_file_structure():
    """测试结果文件结构"""
    print("=" * 60)
    print("测试6: 结果文件结构")
    print("=" * 60)
    
    # 创建模拟器
    simulator = XYModelSimulator(
        lattice_size=8,
        equilibrium_steps=500,
        measurement_steps=1000,
        random_seed=111
    )
    
    # 运行模拟
    results = simulator.run_simulation(
        temperature_range=(0.1, 1.0), 
        num_temperatures=5
    )
    
    # 检查主文件夹
    output_dir = simulator.get_output_directory()
    assert os.path.exists(output_dir), f"主文件夹不存在: {output_dir}"
    
    # 检查子文件夹
    figures_dir = os.path.join(output_dir, 'figures')
    spin_dir = os.path.join(output_dir, 'spin_configurations')
    assert os.path.exists(figures_dir), "figures文件夹不存在"
    assert os.path.exists(spin_dir), "spin_configurations文件夹不存在"
    
    # 检查文件
    data_file = os.path.join(output_dir, 'simulation_results.txt')
    config_file = os.path.join(output_dir, 'simulation_config.txt')
    assert os.path.exists(data_file), "数据文件不存在"
    assert os.path.exists(config_file), "配置文件不存在"
    
    # 检查图表文件
    figure_files = [f for f in os.listdir(figures_dir) if f.endswith('.pdf')]
    assert len(figure_files) == 4, f"图表文件数量不正确: {len(figure_files)}"
    
    print("✓ 结果文件结构测试通过")
    print(f"主文件夹: {output_dir}")
    print(f"包含 {len(figure_files)} 个图表文件")
    print(f"包含 {len(os.listdir(spin_dir))} 个自旋图片")
    
    return simulator


def test_error_handling():
    """测试错误处理"""
    print("=" * 60)
    print("测试7: 错误处理")
    print("=" * 60)
    
    # 测试未运行模拟时获取结果
    simulator = XYModelSimulator()
    
    try:
        results = simulator.get_results()
        assert False, "应该抛出错误"
    except ValueError as e:
        assert "未找到模拟结果" in str(e), f"错误信息不正确: {e}"
        print("✓ 错误处理测试通过")
    
    # 测试无效参数
    try:
        simulator.plot_results()
        assert False, "应该抛出错误"
    except ValueError as e:
        assert "未找到模拟结果" in str(e), f"错误信息不正确: {e}"
        print("✓ 错误处理测试通过")
    
    return simulator


def test_performance_comparison():
    """测试不同参数下的性能比较"""
    print("=" * 60)
    print("测试8: 性能比较")
    print("=" * 60)
    
    configs = [
        {'lattice_size': 8, 'measurement_steps': 1000},
        {'lattice_size': 16, 'measurement_steps': 2000},
        {'lattice_size': 8, 'measurement_steps': 2000},
    ]
    
    results = []
    
    for i, config in enumerate(configs):
        print(f"\n配置 {i+1}: {config}")
        
        simulator = XYModelSimulator(
            lattice_size=config['lattice_size'],
            measurement_steps=config['measurement_steps'],
            random_seed=222 + i
        )
        
        start_time = time.time()
        result = simulator.run_simulation(
            temperature_range=(0.1, 1.5), 
            num_temperatures=5
        )
        elapsed_time = time.time() - start_time
        
        timing = result['timing']
        
        results.append({
            'config': config,
            'total_time': elapsed_time,
            'avg_time_per_temp': timing['avg_time_per_temp'],
            'output_dir': simulator.get_output_directory()
        })
        
        print(f"  总耗时: {elapsed_time:.2f}秒")
        print(f"  平均每温度点: {timing['avg_time_per_temp']:.2f}秒")
    
    print("\n✓ 性能比较测试完成")
    return results


def main():
    """主测试函数"""
    print("XY模型模拟器测试开始")
    print("=" * 80)
    
    try:
        # 运行所有测试
        test_basic_simulation()
        test_gpu_acceleration()
        test_progress_display()
        test_multiple_runs()
        test_spin_visualization()
        test_file_structure()
        test_error_handling()
        test_performance_comparison()
        
        print("\n" + "=" * 80)
        print("所有测试通过！")
        print("XY模型模拟器功能正常。")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()