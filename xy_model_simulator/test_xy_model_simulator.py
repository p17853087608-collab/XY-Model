"""
Test suite for XYModelSimulator with performance tests

This script contains comprehensive tests for XYModelSimulator class,
including unit tests, integration tests, and performance comparisons.
"""

import unittest
import numpy as np
import os
import sys
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端进行测试
import matplotlib.pyplot as plt
import time

# 将当前工作目录添加到路径以导入模拟器
sys.path.insert(0, os.getcwd())

from xy_model_simulator import XYModelSimulator


class TestXYModelSimulator(unittest.TestCase):
    """XYModelSimulator类的测试用例"""
    
    def setUp(self):
        """每个测试方法前设置测试装置"""
        self.test_dir = tempfile.mkdtemp()
        self.simulator = XYModelSimulator(
            lattice_size=4,           # 用于快速测试的小晶格
            equilibrium_steps=10,     # 用于测试的最小步数
            measurement_steps=50,     # 用于测试的最小步数
            interaction_constant=1.0,
            random_seed=42            # 固定种子以确保可重现性
        )
    
    def tearDown(self):
        """每个测试方法后清理"""
        plt.close('all')  # 关闭所有matplotlib图形
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """测试模拟器初始化"""
        # 测试默认初始化
        default_sim = XYModelSimulator()
        self.assertEqual(default_sim.L, 16)
        self.assertEqual(default_sim.ESTEP, 1000)
        self.assertEqual(default_sim.STEP, 10000)
        self.assertEqual(default_sim.J, 1.0)
        
        # 测试自定义初始化
        custom_sim = XYModelSimulator(
            lattice_size=8,
            equilibrium_steps=500,
            measurement_steps=2000,
            interaction_constant=2.0,
            random_seed=123
        )
        self.assertEqual(custom_sim.L, 8)
        self.assertEqual(custom_sim.ESTEP, 500)
        self.assertEqual(custom_sim.STEP, 2000)
        self.assertEqual(custom_sim.J, 2.0)
        
        # 测试配置存储
        config = custom_sim.get_config()
        self.assertEqual(config['晶格尺寸'], 8)
        self.assertEqual(config['平衡步数'], 500)
        self.assertEqual(config['测量步数'], 2000)
        self.assertEqual(config['相互作用常数'], 2.0)
        self.assertEqual(config['随机种子'], 123)
    
    def test_run_simulation(self):
        """测试完整模拟运行"""
        results = self.simulator.run_simulation(
            temperature_range=(0.5, 1.5),
            num_temperatures=5
        )
        
        # 检查结果结构
        self.assertIn('temperature', results)
        self.assertIn('energy', results)
        self.assertIn('magnetization', results)
        self.assertIn('specific_heat', results)
        self.assertIn('susceptibility', results)
        
        # 检查数组长度
        n_temps = 5
        self.assertEqual(len(results['temperature']), n_temps)
        self.assertEqual(len(results['energy']), n_temps)
        self.assertEqual(len(results['magnetization']), n_temps)
        self.assertEqual(len(results['specific_heat']), n_temps)
        self.assertEqual(len(results['susceptibility']), n_temps)
        
        # 检查温度范围
        self.assertAlmostEqual(results['temperature'][0], 0.5)
        self.assertAlmostEqual(results['temperature'][-1], 1.5)
        
        # 检查物理合理性
        self.assertTrue(np.all(results['energy'] <= 0))
        self.assertTrue(np.all(results['magnetization'] >= 0))
        self.assertTrue(np.all(results['specific_heat'] >= 0))
        self.assertTrue(np.all(results['susceptibility'] >= 0))
    
    def test_spin_visualization(self):
        """测试自旋可视化功能"""
        # 首先运行模拟
        self.simulator.run_simulation(
            temperature_range=(0.5, 1.5),
            num_temperatures=3
        )
        
        # 测试自旋可视化
        self.simulator.generate_spin_visualization(
            output_dir=self.test_dir,
            arrow_density=1
        )
        
        # 检查是否创建了文件
        for idx in range(3):
            temp = self.simulator.spin_configurations[idx]['temperature']
            filename = f'spin_config_T_{temp:.3f}.png'
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"文件 {filename} 未创建")
    
    def test_plot_results(self):
        """测试绘图功能"""
        # 首先运行模拟
        self.simulator.run_simulation(
            temperature_range=(0.5, 1.5),
            num_temperatures=5
        )
        
        # 测试不保存的绘图
        self.simulator.plot_results(save_plots=False, show_plots=False)
        
        # 测试保存的绘图
        self.simulator.plot_results(
            save_plots=True,
            output_dir=self.test_dir,
            show_plots=False,
            file_format='png'
        )
        
        # 检查是否创建了文件
        expected_files = [
            'energy_vs_temperature.png',
            'specific_heat_vs_temperature.png',
            'magnetization_vs_temperature.png',
            'susceptibility_vs_temperature.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"文件 {filename} 未创建")
    
    def test_save_results(self):
        """测试结果保存功能"""
        # 首先运行模拟
        self.simulator.run_simulation(
            temperature_range=(0.5, 1.5),
            num_temperatures=5
        )
        
        # 保存结果
        filename = 'test_results.txt'
        self.simulator.save_results(filename=filename, output_dir=self.test_dir)
        
        # 检查是否创建了文件
        filepath = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(filepath))
        
        # 检查文件内容
        with open(filepath, 'r') as f:
            content = f.read()
            self.assertIn('# Temperature', content)
            self.assertIn('Energy', content)
            self.assertIn('Specific Heat', content)
            self.assertIn('Magnetization', content)
            self.assertIn('Susceptibility', content)
        
        # 测试加载数据
        data = np.loadtxt(filepath, skiprows=1)
        self.assertEqual(data.shape, (5, 5))  # 5个温度，5列
    
    def test_get_results_get_config(self):
        """测试get_results和get_config方法"""
        # 测试模拟前的配置
        config = self.simulator.get_config()
        self.assertIsInstance(config, dict)
        self.assertIn('晶格尺寸', config)
        self.assertIn('平衡步数', config)
        self.assertIn('测量步数', config)
        self.assertIn('相互作用常数', config)
        self.assertIn('随机种子', config)
        
        # 测试模拟前的结果（应该报错）
        with self.assertRaises(ValueError):
            self.simulator.get_results()
        
        # 运行模拟
        self.simulator.run_simulation(
            temperature_range=(0.5, 1.5),
            num_temperatures=3
        )
        
        # 测试模拟后的结果
        results = self.simulator.get_results()
        self.assertIsInstance(results, dict)
        self.assertIn('temperature', results)
        self.assertIn('energy', results)
        self.assertIn('magnetization', results)
        self.assertIn('specific_heat', results)
        self.assertIn('susceptibility', results)
        self.assertIn('config', results)
    
    def test_set_config(self):
        """测试配置更新"""
        # 测试有效的配置更新
        self.simulator.set_config(
            lattice_size=8,
            equilibrium_steps=200,
            measurement_steps=1000,
            interaction_constant=2.0,
            random_seed=123
        )
        
        # 检查是否更新了值
        self.assertEqual(self.simulator.L, 8)
        self.assertEqual(self.simulator.ESTEP, 200)
        self.assertEqual(self.simulator.STEP, 1000)
        self.assertEqual(self.simulator.J, 2.0)
        
        # 检查是否更新了配置
        config = self.simulator.get_config()
        self.assertEqual(config['晶格尺寸'], 8)
        self.assertEqual(config['平衡步数'], 200)
        self.assertEqual(config['测量步数'], 1000)
        self.assertEqual(config['相互作用常数'], 2.0)
        self.assertEqual(config['随机种子'], 123)
        
        # 测试无效的配置参数
        with self.assertRaises(ValueError):
            self.simulator.set_config(invalid_param=42)
        
        # 测试配置更改后是否清除了结果
        self.simulator.run_simulation(temperature_range=(0.5, 1.0), num_temperatures=2)
        self.assertIsNotNone(self.simulator.results)
        
        self.simulator.set_config(lattice_size=4)
        self.assertEqual(self.simulator.results, {})
    
    def test_reproducibility(self):
        """测试固定随机种子的模拟可重现性"""
        # 创建两个具有相同种子的相同模拟器
        sim1 = XYModelSimulator(
            lattice_size=4,
            equilibrium_steps=10,
            measurement_steps=20,
            random_seed=42
        )
        
        sim2 = XYModelSimulator(
            lattice_size=4,
            equilibrium_steps=10,
            measurement_steps=20,
            random_seed=42
        )
        
        # 运行相同的模拟
        results1 = sim1.run_simulation(temperature_range=(1.0, 1.5), num_temperatures=3)
        results2 = sim2.run_simulation(temperature_range=(1.0, 1.5), num_temperatures=3)
        
        # 检查结果是否相同
        np.testing.assert_array_almost_equal(results1['temperature'], results2['temperature'])
        np.testing.assert_array_almost_equal(results1['energy'], results2['energy'])
        np.testing.assert_array_almost_equal(results1['magnetization'], results2['magnetization'])
        np.testing.assert_array_almost_equal(results1['specific_heat'], results2['specific_heat'])
        np.testing.assert_array_almost_equal(results1['susceptibility'], results2['susceptibility'])
    
    def test_physical_consistency(self):
        """测试结果的物理一致性"""
        # 运行具有更多点的模拟以获得更好的统计
        self.simulator.set_config(
            lattice_size=6,
            equilibrium_steps=50,
            measurement_steps=200
        )
        
        results = self.simulator.run_simulation(
            temperature_range=(0.1, 2.0),
            num_temperatures=10
        )
        
        temperatures = results['temperature']
        energies = results['energy']
        magnetizations = results['magnetization']
        
        # 测试温度排序
        self.assertTrue(np.all(np.diff(temperatures) > 0))
        
        # 测试能量行为（应该随温度增加）
        # 注意：这是总体趋势，不一定是单调的
        energy_low_T = np.mean(energies[:3])  # 低温平均
        energy_high_T = np.mean(energies[-3:])  # 高温平均
        self.assertGreater(energy_high_T, energy_low_T)
        
        # 测试磁化强度行为（应该随温度降低）
        mag_low_T = np.mean(magnetizations[:3])
        mag_high_T = np.mean(magnetizations[-3:])
        self.assertGreater(mag_low_T, mag_high_T)
        
        # 测试值范围
        self.assertTrue(np.all(energies <= 0))
        self.assertTrue(np.all(magnetizations >= 0))
        self.assertTrue(np.all(magnetizations <= 1))
        self.assertTrue(np.all(results['specific_heat'] >= 0))
        self.assertTrue(np.all(results['susceptibility'] >= 0))


class TestXYModelSimulatorPerformance(unittest.TestCase):
    """XYModelSimulator的性能测试类"""
    
    def setUp(self):
        """设置性能测试"""
        self.test_dir = tempfile.mkdtemp()
        
        # 创建不同尺寸的模拟器用于性能比较
        self.small_sim = XYModelSimulator(
            lattice_size=8,
            equilibrium_steps=100,
            measurement_steps=500,
            random_seed=42
        )
        
        self.medium_sim = XYModelSimulator(
            lattice_size=16,
            equilibrium_steps=500,
            measurement_steps=2000,
            random_seed=42
        )
        
        self.large_sim = XYModelSimulator(
            lattice_size=32,
            equilibrium_steps=1000,
            measurement_steps=5000,
            random_seed=42
        )
    
    def tearDown(self):
        """清理性能测试"""
        plt.close('all')
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def measure_simulation_time(self, simulator, num_runs=5):
        """测量模拟运行时间"""
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            simulator.run_simulation(
                temperature_range=(0.5, 2.5),
                num_temperatures=10
            )
            end_time = time.time()
            times.append(end_time - start_time)
        
        return np.mean(times), np.std(times)
    
    def test_performance_comparison(self):
        """测试不同晶格尺寸的性能"""
        print("\n性能测试结果:")
        print("=" * 50)
        
        # 测试小尺寸晶格
        mean_time_small, std_time_small = self.measure_simulation_time(self.small_sim)
        print(f"小晶格 (8x8): 平均时间 = {mean_time_small:.3f} ± {std_time_small:.3f} 秒")
        
        # 测试中等尺寸晶格
        mean_time_medium, std_time_medium = self.measure_simulation_time(self.medium_sim)
        print(f"中等晶格 (16x16): 平均时间 = {mean_time_medium:.3f} ± {std_time_medium:.3f} 秒")
        
        # 测试大尺寸晶格
        mean_time_large, std_time_large = self.measure_simulation_time(self.large_sim)
        print(f"大晶格 (32x32): 平均时间 = {mean_time_large:.3f} ± {std_time_large:.3f} 秒")
        
        # 计算缩放关系
        size_ratio_medium_small = (16/8)**2
        time_ratio_medium_small = mean_time_medium / mean_time_small
        print(f"中等/小尺寸时间比: {time_ratio_medium_small:.2f} (理论值: {(16/8)**2:.2f})")
        
        size_ratio_large_medium = (32/16)**2
        time_ratio_large_medium = mean_time_large / mean_time_medium
        print(f"大/中等尺寸时间比: {time_ratio_large_medium:.2f} (理论值: {(32/16)**2:.2f})")
        
        size_ratio_large_small = (32/8)**2
        time_ratio_large_small = mean_time_large / mean_time_small
        print(f"大/小尺寸时间比: {time_ratio_large_small:.2f} (理论值: {(32/8)**2:.2f})")
        
        print("=" * 50)
        
        # 验证时间复杂度接近理论预期
        # 理论时间复杂度: O(L^2 × steps × temperatures)
        # 预期时间比应该接近尺寸比的平方
        expected_ratio_medium_small = size_ratio_medium_small
        expected_ratio_large_small = size_ratio_large_small
        
        # 允许一定的误差范围（由于常数因子和其他开销）
        tolerance = 0.3
        
        self.assertLess(abs(time_ratio_medium_small - expected_ratio_medium_small) / expected_ratio_medium_small, tolerance)
        self.assertLess(abs(time_ratio_large_small - expected_ratio_large_small) / expected_ratio_large_small, tolerance)
    
    def test_memory_usage(self):
        """测试内存使用情况"""
        import psutil
        import os
        
        print("\n内存使用测试:")
        print("=" * 50)
        
        # 测试不同晶格尺寸的内存使用
        for size, name in [(8, "小"), (16, "中等"), (32, "大")]:
            # 创建模拟器
            sim = XYModelSimulator(
                lattice_size=size,
                equilibrium_steps=100,
                measurement_steps=500,
                random_seed=42
            )
            
            # 获取进程内存使用
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # 运行模拟
            sim.run_simulation(
                temperature_range=(0.5, 1.5),
                num_temperatures=5
            )
            
            memory_after = process.memory_info().rss / (1024 * 1024)  # MB
            memory_used = memory_after - memory_before
            
            print(f"{name}晶格 ({size}x{size}): 内存使用 = {memory_used:.2f} MB")
        
        print("=" * 50)
    
    def test_optimization_effectiveness(self):
        """测试优化效果"""
        print("\n优化效果测试:")
        print("=" * 50)
        
        # 创建未优化的模拟器（如果可能的话）
        try:
            # 尝试导入未优化的版本
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'unoptimized'))
            from xy_model_simulator_unoptimized import XYModelSimulator as UnoptimizedSimulator
            
            unoptimized_sim = UnoptimizedSimulator(
                lattice_size=16,
                equilibrium_steps=500,
                measurement_steps=2000,
                random_seed=42
            )
            
            # 测量未优化版本的时间
            mean_time_unopt, _ = self.measure_simulation_time(unoptimized_sim, num_runs=3)
            print(f"未优化版本平均时间: {mean_time_unopt:.3f} 秒")
            
        except ImportError:
            print("未优化版本不可用，跳过对比测试")
            mean_time_unopt = None
        
        # 测试优化版本的时间
        mean_time_opt, _ = self.measure_simulation_time(self.medium_sim, num_runs=3)
        print(f"优化版本平均时间: {mean_time_opt:.3f} 秒")
        
        if mean_time_unopt is not None:
            speedup = mean_time_unopt / mean_time_opt
            print(f"加速比: {speedup:.2f}x")
        
        print("=" * 50)


def run_example_simulation():
    """运行示例模拟并显示结果"""
    print("=" * 60)
    print("XY Model Simulator - Example Simulation (优化版)")
    print("=" * 60)
    
    # 创建模拟器
    simulator = XYModelSimulator(
        lattice_size=8,
        equilibrium_steps=200,
        measurement_steps=1000,
        random_seed=42
    )
    
    print("配置:")
    for key, value in simulator.get_config().items():
        print(f"  {key}: {value}")
    
    print("\n运行模拟...")
    start_time = time.time()
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.0),
        num_temperatures=10
    )
    end_time = time.time()
    
    print(f"\n模拟完成!")
    print(f"运行时间: {end_time - start_time:.2f} 秒")
    print(f"温度范围: {results['temperature'][0]:.2f} - {results['temperature'][-1]:.2f}")
    print(f"能量范围: {results['energy'].min():.4f} - {results['energy'].max():.4f}")
    print(f"磁化强度范围: {results['magnetization'].min():.4f} - {results['magnetization'].max():.4f}")
    
    # 找到比热峰值
    peak_idx = np.argmax(results['specific_heat'])
    peak_temp = results['temperature'][peak_idx]
    peak_cv = results['specific_heat'][peak_idx]
    print(f"比热峰值在 T = {peak_temp:.3f}, Cv = {peak_cv:.6f}")
    
    print("\n示例模拟成功完成!")
    print("=" * 60)


if __name__ == '__main__':
    # 运行示例模拟
    run_example_simulation()
    
    print("\n" + "=" * 60)
    print("运行测试套件")
    print("=" * 60)
    
    # 创建测试套件并运行测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestXYModelSimulator)
    suite.addTests(loader.loadTestsFromTestCase(TestXYModelSimulatorPerformance))
    
    # 运行测试，详细输出性能测试结果
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print(f"\n测试总数: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.failures:
        print("\n失败:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\n错误:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n所有测试通过! ✓")
    else:
        print("\n部分测试失败! ✗")
    
    print("=" * 60)