"""
XY模型并行模拟器 - 主入口类
提供简洁的接口用于运行XY模型并行模拟
"""

from xy_parallel import ParallelSimulationManager
from typing import Optional, Tuple, Dict, Any


class ParallelXYModelSimulator(ParallelSimulationManager):
    """
    XY模型并行模拟器类
    
    使用Swendsen-Wang算法进行蒙特卡洛模拟
    专门针对HPC单节点多核平台优化
    """
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None,
                 num_processes: int = 4):
        """
        初始化并行XY模型模拟器
        
        参数:
            lattice_size: 晶格尺寸(LxL)，默认值: 16
            equilibrium_steps: 系统平衡步数，默认值: 1000
            measurement_steps: 物理量测量步数，默认值: 10000
            interaction_constant: 交换相互作用常数J，默认值: 1.0
            random_seed: 随机种子，默认值: None
            num_processes: 并行进程数，默认值: 4
        """
        super().__init__(
            lattice_size=lattice_size,
            equilibrium_steps=equilibrium_steps,
            measurement_steps=measurement_steps,
            interaction_constant=interaction_constant,
            random_seed=random_seed,
            num_processes=num_processes
        )
        
        # 添加向后兼容的属性
        self.use_gpu = False
        self.device = 'CPU'
    
    def run_simulation(self, temperature_range: Tuple[float, float] = (0.1, 2.5), 
                      num_temperatures: int = 10) -> Dict[str, Any]:
        """
        主模拟入口函数，自动选择并行或串行模式
        
        参数:
            temperature_range: 温度范围
            num_temperatures: 温度点数量
            
        返回:
            模拟结果字典
        """
        # 自动选择最优运行模式
        if num_temperatures >= 4 and self.num_processes > 1:
            print(f"检测到多个温度点({num_temperatures})和多核环境({self.num_processes}核)，自动启用并行模式")
            return self.run_parallel_simulation(temperature_range, num_temperatures)
        else:
            print(f"使用串行模式运行")
            # 这里可以调用原有的串行方法，为了简化直接使用并行方法的单进程版本
            original_processes = self.num_processes
            self.num_processes = 1
            result = self.run_parallel_simulation(temperature_range, num_temperatures)
            self.num_processes = original_processes
            return result


if __name__ == "__main__":
    # 示例用法
    simulator = ParallelXYModelSimulator(
        lattice_size=16,
        equilibrium_steps=2000,
        measurement_steps=100,
        interaction_constant=1.0,
        random_seed=42,
        num_processes=4
    )
    
    # 运行并行模拟
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.5),
        num_temperatures=20
    )
    
    print("并行模拟完成！")
    print(f"结果已保存到: {simulator.base_output_dir}")
