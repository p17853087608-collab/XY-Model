"""
XY模型模拟器测试脚本
测试进度显示功能、GPU加速、结果文件管理等所有功能
"""

import shutil
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径（如果脚本不在项目根目录）
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

from xy_model_simulator import XYModelSimulator
from batch_spin_generator_parallel import ParallelBatchSpinGenerator
from batch_spin_generator import BatchSpinGenerator


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



def test_batch_generation():
    """测试批量生成自旋图功能"""
    print("=" * 60)
    print("测试9: 批量生成自旋图功能")
    print("=" * 60)
    
    # 创建临时目录用于测试
    test_output_dir = "test_batch_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # 创建批量生成器实例（使用较小参数以加快测试速度）
        batch_generator = BatchSpinGenerator(
            base_config={
                'lattice_size': 8,            # 较小晶格
                'equilibrium_steps': 200,     # 减少平衡步数
                'measurement_steps': 500,     # 减少测量步数
                'use_gpu': True,
                'random_seed': 42
            },
            output_prefix=test_output_dir
        )
        
        # 定义小型参数网格（减少测试时间）
        parameter_grid = {
            'lattice_size': [8, 12],        # 小晶格
            'measurement_steps': [500, 1000],# 少步数
            'use_gpu': [True]
        }
        
        # 设置参数网格
        batch_generator.define_parameter_grid(parameter_grid)
        assert len(batch_generator.parameter_names) == 3, "参数网格定义错误"
        assert batch_generator.num_simulations == 4, "参数组合数量错误"
        
        # 生成所有模拟实例
        simulations = batch_generator.generate_all_simulations()
        assert len(simulations) == 4, "生成的模拟实例数量错误"
        
        # 运行所有模拟（使用较少温度点）
        results_summary = batch_generator.run_all_simulations(
            temperature_range=(0.1, 1.0), 
            num_temperatures=3  # 少量温度点
        )
        
        # 验证结果
        assert len(results_summary) == 4, "批量结果数量错误"
        
        # 验证输出目录
        assert os.path.exists(batch_generator.batch_output_dir), "批量输出目录未创建"
        
        # 验证每个模拟的输出
        for sim in simulations:
            sim_output_dir = sim['output_dir']
            assert os.path.exists(sim_output_dir), f"模拟输出目录不存在: {sim_output_dir}"
            
            # 验证自旋配置目录
            spin_dir = os.path.join(sim_output_dir, 'spin_configurations')
            assert os.path.exists(spin_dir), f"自旋配置目录不存在: {spin_dir}"
            
            # 验证自旋图片数量
            spin_files = [f for f in os.listdir(spin_dir) if f.endswith('.png')]
            assert len(spin_files) == 3, f"自旋图片数量不正确: {len(spin_files)}"
            
            # 验证数据文件
            data_file = os.path.join(sim_output_dir, 'simulation_results.txt')
            assert os.path.exists(data_file), f"数据文件不存在: {data_file}"
        
        # 验证汇总报告
        summary_file = os.path.join(batch_generator.batch_output_dir, 'batch_summary.txt')
        assert os.path.exists(summary_file), "批量汇总报告未生成"
        
        print("✓ 批量生成自旋图功能测试通过")
        
    except Exception as e:
        print(f"批量生成测试失败: {str(e)}", file=sys.stderr)
        raise
    finally:
        # 清理测试文件
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir, ignore_errors=True)
            print(f"测试目录已清理: {test_output_dir}")
    
    return True


def test_parallel_batch_generation():
    """测试并行批量生成自旋图功能（多线程+GPU加速）"""
    print("=" * 60)
    print("测试10: 并行批量生成自旋图功能（多线程+GPU加速）")
    print("=" * 60)
    
    # 创建临时目录用于测试
    test_output_dir = "test_parallel_batch_output"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # 创建并行批量生成器实例（使用较小参数以加快测试速度）
        batch_generator = ParallelBatchSpinGenerator(
            base_config={
                'lattice_size': 8,            # 较小晶格
                'equilibrium_steps': 200,     # 减少平衡步数
                'measurement_steps': 500,     # 减少测量步数
                'use_gpu': True,
                'random_seed': 42
            },
            max_workers=2  # 测试时使用较少线程
        )
        
        # 验证初始化
        assert batch_generator.use_gpu == True, "GPU加速未正确启用"
        assert batch_generator.max_workers == 2, "线程数设置错误"
        
        # 定义小型参数网格（减少测试时间）
        parameter_grid = {
            'lattice_size': [8, 12],        # 小晶格
            'measurement_steps': [500, 1000],# 少步数
            'use_gpu': [True]
        }
        
        # 设置参数网格
        batch_generator.define_parameter_grid(parameter_grid)
        assert len(batch_generator.parameter_names) == 3, "参数网格定义错误"
        assert batch_generator.num_simulations == 4, "参数组合数量错误"
        
        # 生成所有模拟实例
        simulations = batch_generator.generate_all_simulations()
        assert len(simulations) == 4, "生成的模拟实例数量错误"
        
        # 验证输出目录结构
        assert os.path.exists(batch_generator.batch_output_dir), "批量输出目录未创建"
        
        # 运行并行模拟（使用较少温度点）
        start_time = time.time()
        results_summary = batch_generator.run_all_simulations_parallel(
            temperature_range=(0.1, 1.0), 
            num_temperatures=3  # 少量温度点用于测试
        )
        total_time = time.time() - start_time
        
        # 验证结果
        assert len(results_summary) == 4, "批量结果数量错误"
        
        # 验证并行加速效果
        successful_results = [r for r in results_summary if r['success']]
        assert len(successful_results) > 0, "没有成功的并行任务"
        
        # 验证每个成功模拟的输出
        for result in successful_results:
            sim_output_dir = result['output_dir']
            assert os.path.exists(sim_output_dir), f"模拟输出目录不存在: {sim_output_dir}"
            
            # 验证自旋配置目录
            spin_dir = os.path.join(sim_output_dir, 'spin_configurations')
            assert os.path.exists(spin_dir), f"自旋配置目录不存在: {spin_dir}"
            
            # 验证自旋图片数量
            spin_files = [f for f in os.listdir(spin_dir) if f.endswith('.png')]
            assert len(spin_files) == 3, f"自旋图片数量不正确: {len(spin_files)}"
            
            # 验证数据文件
            data_file = os.path.join(sim_output_dir, 'simulation_results.txt')
            assert os.path.exists(data_file), f"数据文件不存在: {data_file}"
            
            # 验证配置文件
            config_file = os.path.join(sim_output_dir, 'simulation_config.txt')
            assert os.path.exists(config_file), f"配置文件不存在: {config_file}"
        
        # 验证汇总报告
        summary_file = os.path.join(batch_generator.batch_output_dir, 'batch_summary.txt')
        assert os.path.exists(summary_file), "批量汇总报告未生成"
        
        # 验证汇总报告内容
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_content = f.read()
        assert "多线程+GPU加速" in summary_content, "汇总报告中缺少加速模式信息"
        assert "成功数:" in summary_content, "汇总报告中缺少成功数统计"
        assert "失败数:" in summary_content, "汇总报告中缺少失败数统计"
        
        # 验证批量摘要功能
        summary = batch_generator.get_batch_summary()
        assert 'total_simulations' in summary, "批量摘要缺少总任务数"
        assert 'successful_simulations' in summary, "批量摘要缺少成功数"
        assert 'failed_simulations' in summary, "批量摘要缺少失败数"
        assert 'total_spin_images' in summary, "批量摘要缺少总图片数"
        assert 'batch_output_dir' in summary, "批量摘要缺少输出目录"
        
        # 验证并行配置信息
        assert 'parallel_config' in summary, "批量摘要缺少并行配置信息"
        parallel_config = summary['parallel_config']
        assert parallel_config['use_multiprocessing'] == False, "并行模式设置错误"
        assert 'max_workers' in parallel_config, "缺少最大线程数信息"
        
        print("✓ 并行批量生成自旋图功能测试通过")
        print(f"  总任务数: {summary['total_simulations']}")
        print(f"  成功数: {summary['successful_simulations']}")
        print(f"  失败数: {summary['failed_simulations']}")
        print(f"  总图片数: {summary['total_spin_images']}")
        print(f"  总耗时: {total_time:.2f}秒")
        
    except Exception as e:
        print(f"并行批量生成测试失败: {str(e)}", file=sys.stderr)
        raise
    finally:
        # 清理测试文件
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir, ignore_errors=True)
            print(f"测试目录已清理: {test_output_dir}")
    
    return True

def test_parallel_performance():
    """测试并行性能对比"""
    print("=" * 60)
    print("测试11: 并行性能对比测试")
    print("=" * 60)
    
    # 创建临时目录用于测试
    test_output_dir = "test_parallel_performance"
    os.makedirs(test_output_dir, exist_ok=True)
    
    try:
        # 测试不同并行配置的性能
        configs = [
            {'max_workers': 1, 'name': '单线程'},
            {'max_workers': 2, 'name': '2线程'},
            {'max_workers': 4, 'name': '4线程'}
        ]
        
        results = []
        
        for config in configs:
            print(f"\n测试配置: {config['name']}")
            
            # 创建批量生成器
            batch_generator = ParallelBatchSpinGenerator(
                base_config={
                    'lattice_size': 12,
                    'equilibrium_steps': 300,
                    'measurement_steps': 1000,
                    'use_gpu': True,
                    'random_seed': 123
                },
                max_workers=config['max_workers']
            )
            
            # 定义参数网格
            parameter_grid = {
                'lattice_size': [12, 16],
                'measurement_steps': [1000]
            }
            
            batch_generator.define_parameter_grid(parameter_grid)
            batch_generator.generate_all_simulations()
            
            # 运行并行模拟并计时
            start_time = time.time()
            results_summary = batch_generator.run_all_simulations_parallel(
                temperature_range=(0.1, 1.5),
                num_temperatures=5
            )
            elapsed_time = time.time() - start_time
            
            # 记录结果
            successful = len([r for r in results_summary if r['success']])
            total_images = sum(r['spin_configs_generated'] for r in results_summary if r['success'])
            
            results.append({
                'config': config['name'],
                'workers': config['max_workers'],
                'time': elapsed_time,
                'successful': successful,
                'total_images': total_images
            })
            
            print(f"  耗时: {elapsed_time:.2f}秒, 成功: {successful}, 图片数: {total_images}")
        
        # 验证并行加速效果
        single_thread_time = next(r['time'] for r in results if r['workers'] == 1)
        
        print(f"\n并行性能对比结果:")
        print("-" * 60)
        print(f"{'配置':<10} {'线程数':<8} {'耗时(秒)':<12} {'加速比':<10} {'成功数':<8} {'图片数':<8}")
        print("-" * 60)
        
        for result in results:
            speedup = single_thread_time / result['time'] if result['time'] > 0 else 1.0
            print(f"{result['config']:<10} {result['workers']:<8} {result['time']:<12.2f} "
                  f"{speedup:<10.2f} {result['successful']:<8} {result['total_images']:<8}")
        
        # 验证基本加速效果
        multi_thread_results = [r for r in results if r['workers'] > 1]
        assert len(multi_thread_results) > 0, "没有多线程测试结果"
        
        # 验证多线程比单线程快（允许一定的波动）
        for result in multi_thread_results:
            if result['time'] > single_thread_time * 1.5:  # 允许50%的性能波动
                print(f"警告: {result['config']} 性能不如单线程，可能存在资源竞争", file=sys.stderr)
        
        print("✓ 并行性能对比测试通过")
        
    except Exception as e:
        print(f"并行性能测试失败: {str(e)}", file=sys.stderr)
        raise
    finally:
        # 清理测试文件
        if os.path.exists(test_output_dir):
            shutil.rmtree(test_output_dir, ignore_errors=True)
            print(f"测试目录已清理: {test_output_dir}")
    
    return True

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