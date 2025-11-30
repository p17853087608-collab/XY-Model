"""
XY模型模拟结果绘图工具

该脚本从XY模型模拟结果中提取数据，生成温度与各物理量的关系图，
包括能量、磁化强度、比热和磁化率。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional, List


class XYPlotter:
    """XY模型模拟结果绘图器"""
    
    def __init__(self, results: Optional[Dict[str, np.ndarray]] = None):
        """
        初始化绘图器
        
        参数:
            results: 包含模拟结果的字典，应包含以下键:
                     - 'temperature': 温度数组
                     - 'energy': 能量数组
                     - 'magnetization': 磁化强度数组
                     - 'susceptibility': 磁化率数组
                     - 'specific_heat': 比热数组
        """
        self.results = results if results is not None else {}
        self.figures_dir = "figures"
        
        # 创建输出目录
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def load_results(self, file_path: str) -> bool:
        """
        从文本文件加载模拟结果
        
        参数:
            file_path: 结果文件路径
            
        返回:
            加载成功返回True，否则返回False
        """
        try:
            # 读取数据（跳过注释行）
            data = np.loadtxt(file_path, comments='#', delimiter='\t')
            
            # 假设数据列顺序: Temperature, Energy, Specific Heat, Magnetization, Susceptibility
            self.results = {
                'temperature': data[:, 0],
                'energy': data[:, 1],
                'specific_heat': data[:, 2],
                'magnetization': data[:, 3],
                'susceptibility': data[:, 4]
            }
            return True
        except Exception as e:
            print(f"加载结果失败: {e}", file=__import__('sys').stderr)
            return False
    
    def set_results(self, results: Dict[str, np.ndarray]) -> None:
        """直接设置结果数据"""
        required_keys = ['temperature', 'energy', 'magnetization', 'susceptibility', 'specific_heat']
        if not all(key in results for key in required_keys):
            raise ValueError("结果数据必须包含所有必要的物理量")
        self.results = results
    
    def plot_all(self, file_format: str = 'pdf') -> List[str]:
        """
        绘制所有物理量与温度的关系图
        
        参数:
            file_format: 输出文件格式，如'pdf', 'png', 'svg'
            
        返回:
            生成的文件路径列表
        """
        if not self.results:
            raise ValueError("未找到结果数据，请先加载或设置结果")
            
        plotted_files = []
        
        # 绘制能量图
        plotted_files.append(self.plot_energy(file_format))
        
        # 绘制磁化强度图
        plotted_files.append(self.plot_magnetization(file_format))
        
        # 绘制比热图
        plotted_files.append(self.plot_specific_heat(file_format))
        
        # 绘制磁化率图
        plotted_files.append(self.plot_susceptibility(file_format))
        
        return plotted_files
    
    def plot_energy(self, file_format: str = 'pdf') -> str:
        """绘制能量随温度变化图"""
        t = self.results['temperature']
        energy = self.results['energy']
        
        plt.figure()
        plt.plot(t, energy, 'r-', linewidth=2)  # 红色实线，无X标记
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Energy per Site $(J)$', fontsize=12)
        plt.title('Energy vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        filename = f'energy_vs_temperature.{file_format}'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, format=file_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filepath
    
    def plot_magnetization(self, file_format: str = 'pdf') -> str:
        """绘制磁化强度随温度变化图"""
        t = self.results['temperature']
        magnetization = self.results['magnetization']
        
        plt.figure()
        plt.plot(t, magnetization, 'b-', linewidth=2)  # 蓝色实线，无X标记
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Magnetization per Site', fontsize=12)
        plt.title('Magnetization vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        filename = f'magnetization_vs_temperature.{file_format}'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, format=file_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filepath
    
    def plot_specific_heat(self, file_format: str = 'pdf') -> str:
        """绘制比热随温度变化图"""
        t = self.results['temperature']
        specific_heat = self.results['specific_heat']
        
        plt.figure()
        plt.plot(t, specific_heat, 'k-', linewidth=2)  # 黑色实线，无X标记
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Specific Heat per Site $(k_B)$', fontsize=12)
        plt.title('Specific Heat vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        filename = f'specific_heat_vs_temperature.{file_format}'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, format=file_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filepath
    
    def plot_susceptibility(self, file_format: str = 'pdf') -> str:
        """绘制磁化率随温度变化图"""
        t = self.results['temperature']
        susceptibility = self.results['susceptibility']
        
        plt.figure()
        plt.plot(t, susceptibility, 'g-', linewidth=2)  # 绿色实线，无X标记
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Magnetic Susceptibility $(k_B/J)$', fontsize=12)
        plt.title('Susceptibility vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        filename = f'susceptibility_vs_temperature.{file_format}'
        filepath = os.path.join(self.figures_dir, filename)
        plt.savefig(filepath, format=file_format, bbox_inches='tight', dpi=300)
        plt.close()
        
        return filepath


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XY模型模拟结果绘图工具')
    parser.add_argument('-i', '--input', help='模拟结果文件路径', required=True)
    parser.add_argument('-f', '--format', help='输出文件格式(pdf/png/svg)', default='pdf')
    
    args = parser.parse_args()
    
    # 创建绘图器实例
    plotter = XYPlotter()
    
    # 加载结果并绘图
    if plotter.load_results(args.input):
        files = plotter.plot_all(args.format)
        print(f"绘图完成，生成文件:")
        for file in files:
            print(f" - {file}")


if __name__ == "__main__":
    main()