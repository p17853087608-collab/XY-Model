import os
import sys
import time
import numpy as np

# 添加当前目录到系统路径，确保能导入xy_model_simulator
try:
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # 如果__file__未定义（如某些执行环境），使用当前工作目录
    current_dir = os.getcwd()
sys.path.append(current_dir)

from xy_model_simulator import XYModelSimulator

class BatchSpinGenerator(XYModelSimulator):
    """批量生成XY模型自旋图的生成器类，继承自XYModelSimulator"""
    
    def __init__(self, base_config=None, output_prefix="batch_simulation"):
        """
        初始化批量生成器
        
        参数:
            base_config: 基础配置字典，将应用于所有模拟
            output_prefix: 批量模拟结果的前缀名
        """
        # 如果未提供基础配置，使用默认值
        self.base_config = base_config or {
            'lattice_size': 16,
            'equilibrium_steps': 1000,
            'measurement_steps': 5000,
            'use_gpu': True,
            'random_seed': None
        }
        
        self.output_prefix = output_prefix
        self.simulations = []  # 存储所有模拟实例
        self.results_summary = []  # 存储结果摘要
        
        # 创建批量结果主目录
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.batch_output_dir = f"{self.output_prefix}_{timestamp}"
        os.makedirs(self.batch_output_dir, exist_ok=True)
        
        print(f"批量自旋图生成器初始化完成，主目录: {self.batch_output_dir}")

    def define_parameter_grid(self, parameter_grid):
        """
        定义参数网格，用于批量生成不同参数组合的模拟
        
        参数:
            parameter_grid: 参数字典，键为参数名，值为参数列表
        """
        self.parameter_grid = parameter_grid
        self.parameter_names = list(parameter_grid.keys())
        self.num_simulations = np.prod([len(v) for v in parameter_grid.values()])
        
        print(f"已定义参数网格，共 {self.num_simulations} 组参数组合")
        for param, values in parameter_grid.items():
            print(f"  {param}: {len(values)} 个值")

    def generate_all_simulations(self):
        """根据参数网格生成所有模拟实例"""
        if not hasattr(self, 'parameter_grid'):
            raise ValueError("请先调用define_parameter_grid定义参数网格")
            
        # 清空现有模拟
        self.simulations = []
        self.results_summary = []
        
        # 生成参数组合的索引
        param_indices = np.meshgrid(*[range(len(v)) for v in self.parameter_grid.values()])
        param_indices = np.array(param_indices).T.reshape(-1, len(self.parameter_names))
        
        # 创建每个参数组合的模拟实例
        for i, indices in enumerate(param_indices):
            # 创建配置字典
            config = self.base_config.copy()
            
            # 应用当前参数组合
            param_values = {}
            for j, param_name in enumerate(self.parameter_names):
                value_index = indices[j]
                param_value = self.parameter_grid[param_name][value_index]
                config[param_name] = param_value
                param_values[param_name] = param_value
            
            # 创建子目录
            subdir_name = f"sim_{i+1:03d}_{'_'.join([f'{k}={v}' for k, v in param_values.items()])}"
            subdir_path = os.path.join(self.batch_output_dir, subdir_name)
            
            # 创建模拟实例
            simulator = XYModelSimulator(
                lattice_size=config['lattice_size'],
                equilibrium_steps=config['equilibrium_steps'],
                measurement_steps=config['measurement_steps'],
                use_gpu=config['use_gpu'],
                random_seed=config['random_seed'] if config['random_seed'] is not None else np.random.randint(10000)
            )
            
            # 重定向输出目录
            simulator.base_output_dir = subdir_path
            simulator.figures_dir = os.path.join(subdir_path, "figures")
            simulator.spin_dir = os.path.join(subdir_path, "spin_configurations")
            
            # 创建目录
            os.makedirs(simulator.base_output_dir, exist_ok=True)
            os.makedirs(simulator.figures_dir, exist_ok=True)
            os.makedirs(simulator.spin_dir, exist_ok=True)
            
            # 存储模拟实例和参数信息
            self.simulations.append({
                'id': i+1,
                'simulator': simulator,
                'parameters': param_values,
                'output_dir': subdir_path
            })
            
        print(f"已创建 {len(self.simulations)} 个模拟实例")
        return self.simulations

    def run_all_simulations(self, temperature_range=(0.1, 2.5), num_temperatures=10):
        """
        运行所有模拟实例
        
        参数:
            temperature_range: 温度范围元组 (min, max)
            num_temperatures: 温度点数量
        """
        if not self.simulations:
            raise ValueError("请先调用generate_all_simulations生成模拟实例")
            
        total_start_time = time.time()
        
        # 运行每个模拟
        for sim in self.simulations:
            sim_id = sim['id']
            simulator = sim['simulator']
            params = sim['parameters']
            
            print(f"\n{'='*80}")
            print(f"运行模拟 {sim_id}/{len(self.simulations)}")
            print(f"参数: {params}")
            print(f"输出目录: {sim['output_dir']}")
            print(f"{'='*80}")
            
            # 运行模拟
            start_time = time.time()
            results = simulator.run_simulation(
                temperature_range=temperature_range,
                num_temperatures=num_temperatures
            )
            elapsed_time = time.time() - start_time
            
            # 记录结果摘要
            self.results_summary.append({
                'simulation_id': sim_id,
                'parameters': params,
                'output_dir': sim['output_dir'],
                'total_time': elapsed_time,
                'temperature_points': len(results['temperature']),
                'spin_configs_generated': len(simulator.spin_configurations)
            })
            
            print(f"\n模拟 {sim_id} 完成，耗时: {elapsed_time:.2f}秒")
        
        # 生成批量结果汇总报告
        self._generate_summary_report()
        
        total_elapsed = time.time() - total_start_time
        print(f"\n{'='*80}")
        print(f"所有 {len(self.simulations)} 个模拟完成，总耗时: {total_elapsed:.2f}秒")
        print(f"批量结果主目录: {self.batch_output_dir}")
        print(f"结果汇总报告: {os.path.join(self.batch_output_dir, 'batch_summary.txt')}")
        print(f"{'='*80}")
        
        return self.results_summary

    def _generate_summary_report(self):
        """生成批量模拟结果汇总报告"""
        report_path = os.path.join(self.batch_output_dir, 'batch_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("XY模型批量模拟结果汇总\n")
            f.write("========================\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总模拟数: {len(self.simulations)}\n")
            f.write(f"基础配置: {self.base_config}\n\n")
            
            # 参数网格信息
            f.write("参数网格:\n")
            for param, values in self.parameter_grid.items():
                f.write(f"  {param}: {values}\n")
            f.write("\n")
            
            # 各模拟结果摘要
            f.write("模拟结果摘要:\n")
            f.write("-"*60 + "\n")
            f.write(f"{'ID':<6} {'参数':<30} {'温度点':<8} {'耗时(秒)':<10} {'输出目录'}\n")
            f.write("-"*60 + "\n")
            
            for summary in self.results_summary:
                param_str = ", ".join([f"{k}={v}" for k, v in summary['parameters'].items()])
                param_str = param_str[:27] + "..." if len(param_str) > 30 else param_str
                
                f.write(f"{summary['simulation_id']:<6} {param_str:<30} {summary['temperature_points']:<8} "
                        f"{summary['total_time']:<10.2f} {os.path.basename(summary['output_dir'])}\n")

    def get_batch_summary(self):
        """获取批量模拟摘要信息"""
        return {
            'total_simulations': len(self.simulations),
            'parameter_grid': self.parameter_grid,
            'base_config': self.base_config,
            'batch_output_dir': self.batch_output_dir,
            'results_summary': self.results_summary
        }

# 示例使用
if __name__ == "__main__":
    # 创建批量生成器实例
    batch_generator = BatchSpinGenerator(
        base_config={
            'lattice_size': 16,
            'equilibrium_steps': 1000,
            'measurement_steps': 5000,
            'use_gpu': True,
            'random_seed': None
        },
        output_prefix="xy_spin_batch"
    )
    
    # 定义参数网格
    parameter_grid = {
        'lattice_size': [16, 24],  # 不同晶格大小
        'measurement_steps': [3000, 5000],  # 不同测量步数
        'use_gpu': [True]  # GPU加速
    }
    
    # 设置参数网格
    batch_generator.define_parameter_grid(parameter_grid)
    
    # 生成所有模拟实例
    batch_generator.generate_all_simulations()
    
    # 运行所有模拟
    batch_generator.run_all_simulations(
        temperature_range=(0.1, 3.0),
        num_temperatures=15
    )