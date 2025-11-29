import os
import sys
import time
import numpy as np
import threading
from queue import Queue
import concurrent.futures

# 添加当前目录到系统路径
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
sys.path.append(current_dir)

from xy_model_simulator import XYModelSimulator

class ParallelBatchSpinGenerator:
    """并行批量生成XY模型自旋图的生成器类，采用多线程+GPU加速方案"""
    
    def __init__(self, base_config=None, output_prefix="batch_simulation", 
                 max_workers=None):
        """
        初始化并行批量生成器
        
        参数:
            base_config: 基础配置字典
            output_prefix: 结果目录前缀
            max_workers: 最大并行线程数
        """
        # 基础配置
        self.base_config = base_config or {
            'lattice_size': 16,
            'equilibrium_steps': 1000,
            'measurement_steps': 5000,
            'use_gpu': True,
            'random_seed': None
        }
        
        # 强制启用GPU加速以提高多线程效率
        if not self.base_config.get('use_gpu', True):
            print("警告：为提高多线程效率，已自动启用GPU加速", file=sys.stderr)
            self.base_config['use_gpu'] = True
        
        # 并行配置（使用多线程+GPU加速）
        self.max_workers = max_workers or min(8, os.cpu_count() + 1)  # 控制线程数避免GPU资源竞争
        
        # 任务状态
        self.simulations = []
        self.results_summary = []
        self.batch_output_dir = ""
        self.progress_lock = threading.Lock()
        self.completed_tasks = 0
        self.total_tasks = 0
        self.start_time = None
        
        print(f"并行批量自旋图生成器初始化完成")
        print(f"计算模式: 多线程+GPU加速")
        print(f"最大并行线程数: {self.max_workers}")
        print(f"GPU加速: 已启用")

    def define_parameter_grid(self, parameter_grid):
        """定义参数网格"""
        self.parameter_grid = parameter_grid
        self.parameter_names = list(parameter_grid.keys())
        self.num_simulations = np.prod([len(v) for v in parameter_grid.values()])
        
        print(f"已定义参数网格，共 {self.num_simulations} 组参数组合")
        for param, values in parameter_grid.items():
            print(f"  {param}: {len(values)} 个值")

    def generate_all_simulations(self):
        """生成所有模拟配置"""
        if not hasattr(self, 'parameter_grid'):
            raise ValueError("请先调用define_parameter_grid定义参数网格")
            
        self.simulations = []
        
        # 创建结果主目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.batch_output_dir = f"xy_spin_batch_{timestamp}"
        os.makedirs(self.batch_output_dir, exist_ok=True)
        
        # 生成参数组合
        param_indices = np.meshgrid(*[range(len(v)) for v in self.parameter_grid.values()])
        param_indices = np.array(param_indices).T.reshape(-1, len(self.parameter_names))
        
        # 创建模拟配置
        for i, indices in enumerate(param_indices):
            config = self.base_config.copy()
            param_values = {}
            
            for j, param_name in enumerate(self.parameter_names):
                value_index = indices[j]
                param_value = self.parameter_grid[param_name][value_index]
                config[param_name] = param_value
                param_values[param_name] = param_value
            
            # 创建子目录
            subdir_name = f"sim_{i+1:03d}_{'_'.join([f'{k}={v}' for k, v in param_values.items()])}"
            subdir_path = os.path.join(self.batch_output_dir, subdir_name)
            
            # 为每个模拟生成独立随机种子
            if config['random_seed'] is None:
                config['random_seed'] = np.random.randint(100000)
            
            self.simulations.append({
                'id': i+1,
                'parameters': param_values,
                'output_dir': subdir_path,
                'config': config
            })
            
        print(f"已创建 {len(self.simulations)} 个模拟配置")
        return self.simulations

    def _run_single_simulation_thread(self, sim_data, temperature_range, num_temperatures):
        """线程安全的单个模拟执行函数"""
        sim_id = sim_data['id']
        params = sim_data['parameters']
        output_dir = sim_data['output_dir']
        config = sim_data['config']
        
        try:
            # 创建模拟器实例
            simulator = XYModelSimulator(
                lattice_size=config['lattice_size'],
                equilibrium_steps=config['equilibrium_steps'],
                measurement_steps=config['measurement_steps'],
                use_gpu=config['use_gpu'],
                random_seed=config['random_seed']
            )
            
            # 设置输出目录
            simulator.base_output_dir = output_dir
            simulator.figures_dir = os.path.join(output_dir, "figures")
            simulator.spin_dir = os.path.join(output_dir, "spin_configurations")
            os.makedirs(simulator.base_output_dir, exist_ok=True)
            os.makedirs(simulator.figures_dir, exist_ok=True)
            os.makedirs(simulator.spin_dir, exist_ok=True)
            
            # 运行模拟（GPU操作会自动异步执行）
            start_time = time.time()
            results = simulator.run_simulation(
                temperature_range=temperature_range,
                num_temperatures=num_temperatures
            )
            elapsed_time = time.time() - start_time
            
            # 更新进度
            with self.progress_lock:
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                elapsed_total = time.time() - self.start_time
                eta = elapsed_total / self.completed_tasks * (self.total_tasks - self.completed_tasks) if self.completed_tasks > 0 else 0
                
                print(f"\r进度: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks}) - "
                      f"模拟 {sim_id} 完成，耗时: {elapsed_time:.2f}秒 - "
                      f"预计剩余: {eta:.1f}秒", end="")
            
            return {
                'simulation_id': sim_id,
                'parameters': params,
                'output_dir': output_dir,
                'total_time': elapsed_time,
                'temperature_points': len(results['temperature']),
                'spin_configs_generated': len(simulator.spin_configurations),
                'success': True,
                'error': None
            }
            
        except Exception as e:
            error_msg = str(e)
            with self.progress_lock:
                self.completed_tasks += 1
                progress = (self.completed_tasks / self.total_tasks) * 100
                print(f"\r进度: {progress:.1f}% ({self.completed_tasks}/{self.total_tasks}) - "
                      f"模拟 {sim_id} 失败: {error_msg[:50]}...", end="")
            
            return {
                'simulation_id': sim_id,
                'parameters': params,
                'output_dir': output_dir,
                'total_time': 0,
                'temperature_points': 0,
                'spin_configs_generated': 0,
                'success': False,
                'error': error_msg
            }

    def run_all_simulations_parallel(self, temperature_range=(0.1, 2.5), num_temperatures=10):
        """并行运行所有模拟（多线程+GPU加速）"""
        if not self.simulations:
            raise ValueError("请先调用generate_all_simulations生成模拟配置")
            
        self.start_time = time.time()
        self.total_tasks = len(self.simulations)
        self.completed_tasks = 0
        
        print(f"\n{'='*80}")
        print(f"开始并行运行 {self.total_tasks} 个模拟任务")
        print(f"温度范围: {temperature_range[0]:.2f} - {temperature_range[1]:.2f}")
        print(f"温度点数: {num_temperatures} (预计生成 {self.total_tasks * num_temperatures} 张自旋图)")
        print(f"结果目录: {self.batch_output_dir}")
        print(f"{'='*80}")
        
        # 使用线程池执行任务（避免进程间通信问题）
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = []
            for sim in self.simulations:
                future = executor.submit(
                    self._run_single_simulation_thread,
                    sim, temperature_range, num_temperatures
                )
                futures.append(future)
            
            # 收集结果
            self.results_summary = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                self.results_summary.append(result)
        
        # 生成汇总报告
        self._generate_summary_report()
        self._print_final_summary()
        
        return self.results_summary

    def _generate_summary_report(self):
        """生成汇总报告"""
        report_path = os.path.join(self.batch_output_dir, 'batch_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("XY模型批量模拟结果汇总\n")
            f.write("========================\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"计算模式: 多线程+GPU加速\n")
            f.write(f"总模拟数: {len(self.simulations)}\n")
            f.write(f"成功数: {sum(1 for r in self.results_summary if r['success'])}\n")
            f.write(f"失败数: {sum(1 for r in self.results_summary if not r['success'])}\n")
            f.write(f"总自旋图片数: {sum(r['spin_configs_generated'] for r in self.results_summary if r['success'])}\n\n")
            
            # 参数网格信息
            f.write("参数网格:\n")
            for param, values in self.parameter_grid.items():
                f.write(f"  {param}: {values}\n")
            
            # 详细结果
            f.write("\n详细结果:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'ID':<6} {'状态':<8} {'参数组合':<30} {'温度点':<8} {'图片数':<8} {'耗时(s)':<8}\n")
            f.write("-"*80 + "\n")
            
            for result in self.results_summary:
                params_str = ", ".join([f"{k}={v}" for k, v in result['parameters'].items()])
                if len(params_str) > 30:
                    params_str = params_str[:27] + "..."
                    
                status = "成功" if result['success'] else "失败"
                f.write(f"{result['simulation_id']:<6} {status:<8} {params_str:<30} "
                        f"{result['temperature_points']:<8} {result['spin_configs_generated']:<8} "
                        f"{result['total_time']:.1f}\n")

    def _print_final_summary(self):
        """打印最终汇总"""
        total = len(self.results_summary)
        success = sum(1 for r in self.results_summary if r['success'])
        failed = total - success
        total_images = sum(r['spin_configs_generated'] for r in self.results_summary if r['success'])
        total_time = sum(r['total_time'] for r in self.results_summary if r['success'])
        elapsed_total = time.time() - self.start_time
        speedup = total_time / elapsed_total if elapsed_total > 0 else 1
        
        print(f"\n{'='*80}")
        print(f"批量模拟完成！")
        print(f"总任务数: {total} 成功: {success} 失败: {failed}")
        print(f"生成自旋图片总数: {total_images}")
        print(f"总计算时间: {elapsed_total:.1f}秒 (并行加速比: {speedup:.1f}x)")
        print(f"结果保存目录: {self.batch_output_dir}")
        print(f"汇总报告: {os.path.join(self.batch_output_dir, 'batch_summary.txt')}")
        print(f"{'='*80}")

# 示例使用
if __name__ == "__main__":
    # 创建并行批量生成器（大规模任务配置）
    batch_generator = ParallelBatchSpinGenerator(
        base_config={
            'lattice_size': 24,
            'equilibrium_steps': 1000,
            'measurement_steps': 5000,
            'use_gpu': True,
            'random_seed': None
        },
        max_workers=4  # 线程数不宜过多，避免GPU资源竞争
    )
    
    # 定义参数网格（生成大量组合）
    parameter_grid = {
        'lattice_size': [16, 24, 32],        # 不同晶格尺寸
        'measurement_steps': [3000, 5000],   # 不同测量步数
        'use_gpu': [True]
    }
    
    # 生成模拟任务
    batch_generator.define_parameter_grid(parameter_grid)
    batch_generator.generate_all_simulations()
    
    # 运行并行模拟（生成数十个温度点的上万张图片）
    batch_generator.run_all_simulations_parallel(
        temperature_range=(0.1, 4.0),  # 温度范围
        num_temperatures=50            # 每个模拟生成50个温度点的自旋图
    )
