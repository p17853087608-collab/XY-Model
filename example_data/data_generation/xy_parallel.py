"""
XY模型模拟器 - 并行处理模块
处理并行模拟、结果保存和报告生成
"""

import os
import time
import datetime
import numpy as np
from typing import Dict, Any, Tuple, Optional
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from xy_simulator_core import XYModelSimulatorCore


def run_single_temperature_parallel(args):
    """
    并行运行单个温度点的函数
    
    参数:
        args: (idx, temperature, config_dict, process_id, task_id)
        
    返回:
        (idx, result, error)
    """
    idx, temperature, config_dict, process_id, task_id = args
    
    try:
        # 为每个进程创建独立的模拟器实例
        simulator = XYModelSimulatorCore(
            lattice_size=config_dict['lattice_size'],
            equilibrium_steps=config_dict['equilibrium_steps'],
            measurement_steps=config_dict['measurement_steps'],
            interaction_constant=config_dict['interaction_constant'],
            random_seed=config_dict['random_seed'],
            process_id=process_id
        )
        
        # 设置任务ID，增加随机性
        simulator.task_id = task_id

        # 运行单个温度点
        result = simulator.run_temperature_simulation(temperature)
        result['original_index'] = idx
        result['process_id'] = process_id
        
        return idx, result, None
        
    except Exception as e:
        print(f"进程 {process_id} 处理温度 {temperature} (索引 {idx}) 时出错: {e}")
        # 返回安全的默认结果
        return idx, {
            'temperature': temperature,
            'energy': 0.0,
            'magnetization': 0.0,
            'susceptibility': 0.0,
            'specific_heat': 0.0,
            'spin_config': np.zeros((config_dict['lattice_size'], config_dict['lattice_size'])),
            'actual_equilibrium_steps': 0,
            'actual_measurements': 0,
            'original_index': idx,
            'process_id': process_id
        }, str(e)


def parallel_progress_monitor(futures, total_temperatures):
    """并行进度监控器"""
    completed = 0
    start_time = time.time()
    
    for future in as_completed(futures):
        try:
            idx, result, error = future.result()
            completed += 1
            
            elapsed = time.time() - start_time
            if completed > 0:
                eta = elapsed * (total_temperatures - completed) / completed
                print(f"进度: {completed}/{total_temperatures} ({completed/total_temperatures*100:.1f}%) "
                      f"- 温度 {result['temperature']:.3f} - ETA: {eta:.1f}秒")
            
        except Exception as e:
            print(f"处理异步结果时出错: {e}")
            completed += 1


class ParallelSimulationManager:
    """并行模拟管理器"""
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None,
                 num_processes: int = 4):
        """
        初始化并行模拟管理器
        
        参数:
            lattice_size: 晶格尺寸
            equilibrium_steps: 平衡步数
            measurement_steps: 测量步数
            interaction_constant: 相互作用常数
            random_seed: 随机种子
            num_processes: 并行进程数
        """
        self.L = lattice_size
        self.lattice_size = lattice_size
        self.ESTEP = equilibrium_steps
        self.STEP = measurement_steps
        self.J = interaction_constant
        self.num_processes = min(num_processes, cpu_count())
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            self.base_seed = random_seed
        else:
            self.base_seed = int(time.time()) % 1000000
        
        # 存储模拟结果
        self.results = {}
        self.timing_data = {}
        self.config = {
            '晶格尺寸': self.L,
            '平衡步数': self.ESTEP,
            '测量步数': self.STEP,
            '相互作用常数': self.J,
            '随机种子': self.base_seed,
            '并行进程数': self.num_processes
        }
        
        # 初始化输出目录变量
        self.base_output_dir = None
        self.spin_dir = None
    
    def run_parallel_simulation(self, temperature_range: Tuple[float, float] = (0.1, 2.5), 
                               num_temperatures: int = 10) -> Dict[str, Any]:
        """
        并行运行多个温度点的XY模型模拟
        
        参数:
            temperature_range: (T_min, T_max)温度范围
            num_temperatures: 温度点数量
            
        返回:
            包含所有温度点模拟结果的字典
        """
        start_time = time.time()
        
        # 预分配所有结果数组
        t_min, t_max = temperature_range
        temperature_array = np.linspace(t_min, t_max, num_temperatures)
        
        magnetization_array = np.zeros(num_temperatures, dtype=np.float64)
        energy_array = np.zeros(num_temperatures, dtype=np.float64)
        susceptibility_array = np.zeros(num_temperatures, dtype=np.float64)
        specific_heat_array = np.zeros(num_temperatures, dtype=np.float64)
        temperature_times = np.zeros(num_temperatures, dtype=np.float64)
        
        # 创建结果目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = np.random.randint(1000, 9999)
        self.base_output_dir = f"parallel_simulation_results_{timestamp}_{random_suffix}"
        
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.spin_dir = os.path.join(self.base_output_dir, "spin_configurations")
        os.makedirs(self.spin_dir, exist_ok=True)
        
        print(f"开始并行XY模型模拟，共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print(f"并行进程数: {self.num_processes}")
        print(f"结果保存目录: {self.base_output_dir}")
        print("=" * 60)
        
        # 准备并行任务
        config_dict = {
            'lattice_size': self.L,
            'equilibrium_steps': self.ESTEP,
            'measurement_steps': self.STEP,
            'interaction_constant': self.J,
            'random_seed': self.base_seed
        }
        
        # 为每次运行生成唯一运行ID
        run_id = int((time.time() * 1000 + os.getpid() * 100) % 1000000)
        
        # 创建增强随机性的任务
        import random
        base_seed_for_tasks = self.base_seed + run_id
        random.seed(base_seed_for_tasks)
        
        # 为每个温度点添加额外的随机偏移
        tasks = []
        for idx, temp in enumerate(temperature_array):
            task_specific_offset = random.randint(0, 999999)
            task_id = (idx * 100000 + run_id + task_specific_offset) % 1000000
            tasks.append((idx, temp, config_dict, idx, task_id))
        
        # 使用ProcessPoolExecutor进行并行计算
        all_results = []
        completed_indices = set()
        
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # 提交所有任务
            futures = {executor.submit(run_single_temperature_parallel, task): task[0] 
                      for task in tasks}
            
            # 监控进度并收集结果
            for future in as_completed(futures):
                try:
                    idx, result, error = future.result()
                    
                    if error is None:
                        all_results.append(result)
                        completed_indices.add(idx)
                        
                        # 立即存储当前结果
                        temperature_times[idx] = time.time() - start_time
                        energy_array[idx] = result['energy']
                        magnetization_array[idx] = result['magnetization']
                        susceptibility_array[idx] = result['susceptibility']
                        specific_heat_array[idx] = result['specific_heat']
                        
                        # 保存自旋配置
                        try:
                            spin_filename = f'spin_config_T_{result["temperature"]:.3f}_raw.npy'
                            spin_path = os.path.join(self.spin_dir, spin_filename)
                            np.save(spin_path, result['spin_config'])
                        except Exception as e:
                            print(f"保存自旋配置时出错: {e}")
                        
                        print(f"完成温度 {result['temperature']:.3f} (进度: {len(completed_indices)}/{num_temperatures})")
                    
                    else:
                        print(f"温度点 {idx} 处理失败: {error}")
                        
                except Exception as e:
                    task_idx = futures[future]
                    print(f"处理温度点 {task_idx} 的异步结果时出错: {e}")
        
        # 按原始顺序排序结果
        all_results.sort(key=lambda x: x['original_index'])
        
        print("=" * 60)
        print("并行模拟完成！正在整理结果...")
        
        # 计算统计信息
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean() if temperature_times.any() else total_time / num_temperatures
        
        # 并行性能统计
        actual_parallel_efficiency = (num_temperatures * avg_time_per_temp) / total_time / self.num_processes * 100
        
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        print(f"并行效率: {actual_parallel_efficiency:.1f}% (理论值: {100.0:.1f}%)")
        
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times,
            'avg_time_per_temp': avg_time_per_temp,
            'min_time': temperature_times.min() if temperature_times.any() else avg_time_per_temp,
            'max_time': temperature_times.max() if temperature_times.any() else avg_time_per_temp,
            'parallel_efficiency': actual_parallel_efficiency,
            'num_processes': self.num_processes
        }
        
        self.results = {
            'temperature': temperature_array,
            'energy': energy_array,
            'magnetization': magnetization_array,
            'specific_heat': specific_heat_array,
            'susceptibility': susceptibility_array,
            'config': self.config.copy(),
            'timing': self.timing_data,
            'detailed_results': all_results
        }
        
        # 保存结果
        try:
            self.save_parallel_results()
            print("并行模拟结果已保存完成")
        except Exception as e:
            print(f"保存结果时出错: {e}")
        
        return self.results
    
    def _convert_numpy_types(self, obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def save_parallel_results(self):
        """保存并行模拟结果"""
        try:
            # 保存主要结果数据
            results_file = os.path.join(self.base_output_dir, "parallel_results.npz")
            np.savez_compressed(results_file,
                              temperature=self.results['temperature'],
                              energy=self.results['energy'],
                              magnetization=self.results['magnetization'],
                              specific_heat=self.results['specific_heat'],
                              susceptibility=self.results['susceptibility'])
            
            # 保存配置和时间统计
            config_file = os.path.join(self.base_output_dir, "simulation_config.json")
            import json
            with open(config_file, 'w', encoding='utf-8') as f:
                config_data = {
                    'config': self._convert_numpy_types(self.config),
                    'timing': self._convert_numpy_types(self.timing_data),
                    'performance_stats': {
                        'total_temperatures': int(len(self.results['temperature'])),
                        'parallel_processes': int(self.num_processes),
                        'avg_time_per_temp': float(self.timing_data['avg_time_per_temp']),
                        'parallel_efficiency': float(self.timing_data.get('parallel_efficiency', 0))
                    }
                }
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 保存详细结果到文本文件
            results_text_file = os.path.join(self.base_output_dir, "results_summary.txt")
            header = "# Temperature\tEnergy\tMagnetization\tSusceptibility\tSpecific_Heat"
            data = np.column_stack((
                self.results['temperature'],
                self.results['energy'], 
                self.results['magnetization'],
                self.results['susceptibility'],
                self.results['specific_heat']
            ))
            np.savetxt(results_text_file, data, header=header, delimiter='\t', fmt='%.6f')
            
            print(f"结果文件已保存:")
            print(f"  - 主要数据: {results_file}")
            print(f"  - 配置文件: {config_file}")
            print(f"  - 结果摘要: {results_text_file}")
            
            # 生成分析报告
            self._generate_analysis_report()
            
        except Exception as e:
            print(f"保存并行结果时出错: {e}")
    
    def _generate_analysis_report(self):
        """生成详细的分析报告"""
        try:
            # 创建分析报告文件
            report_file = os.path.join(self.base_output_dir, "analysis_report.txt")
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("XY模型并行模拟分析报告\n")
                f.write("=" * 50 + "\n\n")
                
                # 基本信息
                f.write("模拟参数:\n")
                f.write(f"  晶格大小: {self.L}×{self.L}\n")
                f.write(f"  平衡步数: {self.ESTEP}\n")
                f.write(f"  测量步数: {self.STEP}\n")
                f.write(f"  相互作用常数: {self.J}\n")
                f.write(f"  并行进程数: {self.num_processes}\n\n")
                
                # 性能统计
                timing = self.timing_data
                f.write("性能统计:\n")
                f.write(f"  总耗时: {timing['total_time']:.2f} 秒\n")
                f.write(f"  平均每温度点耗时: {timing['avg_time_per_temp']:.2f} 秒\n")
                f.write(f"  最快温度点耗时: {timing['min_time']:.2f} 秒\n")
                f.write(f"  最慢温度点耗时: {timing['max_time']:.2f} 秒\n")
                f.write(f"  并行效率: {timing['parallel_efficiency']:.1f}%\n")
                
                if timing['parallel_efficiency'] > 100:
                    theoretical_speedup = self.num_processes
                    actual_speedup = theoretical_speedup * timing['parallel_efficiency'] / 100
                    f.write(f"  理论加速比: {theoretical_speedup:.1f}x\n")
                    f.write(f"  实际加速比: {actual_speedup:.1f}x\n")
                f.write("\n")
                
                # 物理结果分析
                temps = self.results['temperature']
                energy = self.results['energy']
                mag = self.results['magnetization']
                sus = self.results['susceptibility']
                heat = self.results['specific_heat']
                
                f.write("物理结果分析:\n")
                f.write(f"  温度范围: {temps.min():.3f} - {temps.max():.3f}\n")
                f.write(f"  能量范围: {energy.min():.6f} - {energy.max():.6f}\n")
                f.write(f"  磁化强度范围: {mag.min():.6f} - {mag.max():.6f}\n")
                f.write(f"  磁化率峰值: {sus.max():.6f} (在T={temps[sus.argmax()]:.3f})\n")
                f.write(f"  比热峰值: {heat.max():.6f} (在T={temps[heat.argmax()]:.3f})\n")
                
                # 临界温度估计
                tc_sus = temps[sus.argmax()]
                tc_heat = temps[heat.argmax()]
                tc_estimate = (tc_sus + tc_heat) / 2
                
                f.write("\n临界温度分析:\n")
                f.write(f"  磁化率峰值法: Tc = {tc_sus:.3f}\n")
                f.write(f"  比热峰值法: Tc = {tc_heat:.3f}\n")
                f.write(f"  平均估计: Tc = {tc_estimate:.3f}\n")
                f.write(f"  (理论值: Tc ≈ {self.J * 0.89:.3f})\n\n")
                
                # 相变分析
                f.write("相变分析:\n")
                high_temp_mag = np.mean(mag[temps > (tc_estimate + 0.3)])
                low_temp_mag = np.mean(mag[temps < (tc_estimate - 0.3)])
                
                if high_temp_mag < 0.1 and low_temp_mag > 0.5:
                    f.write("  ✓ 观察到明显的有序-无序相变\n")
                elif low_temp_mag > 0.3:
                    f.write("  ⚠ 可能存在部分有序相\n")
                else:
                    f.write("  ✗ 相变特征不明显\n")
                
                f.write(f"  高温磁化强度: {high_temp_mag:.4f}\n")
                f.write(f"  低温磁化强度: {low_temp_mag:.4f}\n")
            
            print(f"  - 分析报告: {report_file}")

        except Exception as e:
            print(f"生成分析报告时出错: {e}")
