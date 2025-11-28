"""
XY模型批量自旋图生成器（最终设计版）

功能说明：
- 批量生成XY模型自旋配置图像
- 支持多进程并行处理
- 自动温度分类存储
- 提供进度监控和性能统计
- 需要配合XY模型模拟器使用

使用方法：
1. 确保xy_model_simulator.py在相同目录
2. 运行此脚本生成批量自旋图
3. 结果按温度分类存储在指定目录
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --------------------------
# 环境初始化
# --------------------------
def initialize_environment() -> bool:
    """初始化环境，检查必要依赖"""
    try:
        # 检查必要的模块
        required_modules = ['numpy', 'matplotlib', 'tqdm']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"错误: 缺少以下模块: {missing_modules}")
            print("请安装缺失的模块: pip install " + " ".join(missing_modules))
            return False
        
        # 检查XY模型模拟器
        try:
            from xy_model_simulator import XYModelSimulator
            print("XY模型模拟器导入成功")
            return True
        except ImportError as e:
            print(f"错误: 无法导入XY模型模拟器: {e}")
            print("请确保xy_model_simulator.py位于当前目录")
            return False
            
    except Exception as e:
        print(f"环境初始化错误: {e}")
        return False


# --------------------------
# 图像生成函数
# --------------------------
def create_spin_image(spin_config: np.ndarray, temperature: float,
                     magnetization: float, susceptibility: float,
                     output_path: str, dpi: int = 150) -> bool:
    """
    创建自旋配置图像
    
    参数:
        spin_config: 自旋角度矩阵
        temperature: 温度值
        magnetization: 磁化强度
        susceptibility: 磁化率
        output_path: 输出路径
        dpi: 图像分辨率
        
    返回:
        bool: 创建是否成功
    """
    try:
        lattice_size = spin_config.shape[0]
        
        # 创建网格
        x = np.arange(lattice_size)
        y = np.arange(lattice_size)
        X, Y = np.meshgrid(x, y)
        
        # 计算自旋向量
        U = np.cos(spin_config)
        V = np.sin(spin_config)
        
        # 创建RGB彩色图像（0.5度精度）
        hue = (spin_config % (2 * np.pi)) / (2 * np.pi)
        saturation = 0.3 + 0.7 * magnetization
        value = 0.9
        
        # 创建HSV图像并转换为RGB
        hsv_image = np.stack([hue, saturation * np.ones_like(hue), value * np.ones_like(hue)], axis=2)
        from matplotlib.colors import hsv_to_rgb
        rgb_image = hsv_to_rgb(hsv_image)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 显示彩色自旋配置
        ax.imshow(rgb_image, origin='lower', 
                  extent=[-0.5, lattice_size - 0.5, -0.5, lattice_size - 0.5])
        
        # 添加网格线
        ax.set_xticks(range(lattice_size))
        ax.set_yticks(range(lattice_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # 设置图形属性
        ax.set_aspect('equal')
        plt.xlim(-0.5, lattice_size - 0.5)
        plt.ylim(-0.5, lattice_size - 0.5)
        
        # 添加标题和物理量信息
        title_text = f'XY Model Spin Configuration (T = {temperature:.3f})'
        info_text = f'Magnetization: {magnetization:.4f}\nSusceptibility: {susceptibility:.4f}'
        
        plt.title(title_text, fontsize=16, pad=20)
        plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=12)
        
        # 添加颜色条说明
        cbar_text = 'Color represents spin orientation:\nRed: 0°, Yellow: 90°, Green: 180°, Blue: 270°'
        plt.text(0.98, 0.02, cbar_text, transform=ax.transAxes, 
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 fontsize=10)
        
        # 保存图片
        plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        return True
    except Exception as e:
        print(f"创建图像失败: {e}")
        return False


def get_temperature_category(temperature: float) -> str:
    """
    根据温度值确定温度类别
    
    参数:
        temperature: 温度值
        
    返回:
        str: 温度类别
    """
    if temperature < 0.5:
        return 'low_temp'
    elif temperature < 1.5:
        return 'medium_temp'
    else:
        return 'high_temp'


# --------------------------
# 温度处理函数
# --------------------------
def process_temperature(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理单个温度点的图像生成
    
    参数:
        config: 温度配置字典
        
    返回:
        Dict: 处理结果
    """
    try:
        temperature = config['temperature']
        num_images = config['num_images']
        lattice_size = config['lattice_size']
        temp_range = config['temp_range']
        num_temperatures = config['num_temperatures']
        use_gpu = config['use_gpu']
        base_output_dir = config['base_output_dir']
        batch_dir = config['batch_dir']
        
        # 导入XY模型模拟器
        from xy_model_simulator import XYModelSimulator
        
        # 创建模拟器
        simulator = XYModelSimulator(
            lattice_size=lattice_size,
            equilibrium_steps=1000,
            measurement_steps=2000,
            use_gpu=use_gpu,
            random_seed=int(temperature * 1000)
        )
        
        # 运行模拟
        results = simulator.run_simulation(
            temperature_range=temp_range,
            num_temperatures=num_temperatures
        )
        
        # 检查模拟结果
        if 'error' in results:
            return {
                'temperature': temperature,
                'error': results['error'],
                'success': False
            }
        
        # 确定温度类别和输出目录
        temp_category = get_temperature_category(temperature)
        output_dir = os.path.join(base_output_dir, batch_dir, "images", temp_category)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成自旋图
        image_paths = []
        for i in range(num_images):
            # 创建新模拟器以获得不同配置
            sim = XYModelSimulator(
                lattice_size=lattice_size,
                equilibrium_steps=1000,
                measurement_steps=2000,
                use_gpu=use_gpu,
                random_seed=int(temperature * 1000 + i + 1)
            )
            
            # 运行单次模拟
            temp_results = sim.run_simulation(
                temperature_range=(temperature - 0.01, temperature + 0.01),
                num_temperatures=1
            )
            
            # 检查模拟结果
            if 'error' in temp_results:
                return {
                    'temperature': temperature,
                    'error': temp_results['error'],
                    'success': False
                }
            
            # 获取自旋配置和物理量
            spin_config = temp_results['spin_configurations'][0]['spin_config']
            magnetization = temp_results['spin_configurations'][0]['magnetization']
            susceptibility = temp_results['spin_configurations'][0]['susceptibility']
            
            # 生成文件名和路径
            filename = f"spin_T{temperature:.3f}_n{i+1:04d}.png"
            output_path = os.path.join(output_dir, filename)
            
            # 生成自旋图
            success = create_spin_image(
                spin_config, temperature, magnetization, 
                susceptibility, output_path
            )
            
            if success:
                image_paths.append(output_path)
            else:
                return {
                    'temperature': temperature,
                    'error': f"图像创建失败: {filename}",
                    'success': False
                }
        
        return {
            'temperature': temperature,
            'num_images': len(image_paths),
            'image_paths': image_paths,
            'temp_category': temp_category,
            'lattice_size': lattice_size,
            'results': results,
            'success': True
        }
    except Exception as e:
        return {
            'temperature': config.get('temperature', -1),
            'error': f"处理温度点时出错: {str(e)}",
            'success': False
        }


# --------------------------
# 批量生成器类
# --------------------------
class BatchSpinGenerator:
    """
    批量自旋图生成器类
    
    功能：
    - 管理批量生成任务
    - 提供进度监控
    - 生成性能报告
    """
    
    def __init__(self, base_output_dir: str = "batch_spin_images", 
                 use_gpu: bool = True, num_workers: int = None):
        """
        初始化批量生成器
        
        参数:
            base_output_dir: 基础输出目录
            use_gpu: 是否使用GPU加速
            num_workers: 并行进程数
        """
        self.base_output_dir = base_output_dir
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.num_workers = num_workers or self._get_optimal_workers()
        
        # 创建输出目录结构
        self._setup_output_structure()
        
        # 性能统计
        self.performance_stats = {
            'total_images': 0,
            'successful_images': 0,
            'failed_images': 0,
            'total_time': 0.0,
            'avg_time_per_image': 0.0,
            'generation_rate': 0.0,
            'errors': []
        }
        
        print(f"批量自旋图生成器初始化完成")
        print(f"GPU加速: {'启用' if self.use_gpu else '禁用'}")
        print(f"并行进程数: {self.num_workers}")
        print(f"输出目录: {self.output_dir}")
    
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        try:
            import cupy as cp
            # 测试基本GPU操作
            test_array = cp.array([1, 2, 3])
            del test_array
            return True
        except ImportError:
            print("警告: CuPy未安装，将使用CPU模式")
            return False
        except Exception as e:
            print(f"警告: GPU不可用，错误: {e}")
            return False
    
    def _get_optimal_workers(self) -> int:
        """获取最优并行进程数"""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        # Windows系统限制进程数
        if sys.platform.startswith('win'):
            return min(4, cpu_count)
        elif self.use_gpu:
            return min(4, cpu_count)
        else:
            return min(cpu_count, 8)
    
    def _setup_output_structure(self):
        """设置输出目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_dir = f"batch_{timestamp}"
        self.output_dir = os.path.join(self.base_output_dir, self.batch_dir)
        self.image_dir = os.path.join(self.output_dir, "images")
        self.data_dir = os.path.join(self.output_dir, "data")
        
        # 创建所有目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        for temp_category in ['low_temp', 'medium_temp', 'high_temp']:
            os.makedirs(os.path.join(self.image_dir, temp_category), exist_ok=True)
    
    def generate_batch_parallel(self, batch_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        并行生成多个温度点的自旋图批次
        
        参数:
            batch_configs: 批次配置列表
            
        返回:
            Dict: 生成结果
        """
        start_time = time.time()
        total_images = sum(config['num_images'] for config in batch_configs)
        
        print(f"开始批量自旋图生成")
        print(f"总图像数量: {total_images}")
        print(f"温度点数量: {len(batch_configs)}")
        print(f"并行进程数: {self.num_workers}")
        print("=" * 60)
        
        # 准备配置参数
        processed_configs = []
        for config in batch_configs:
            processed_config = config.copy()
            processed_config.update({
                'use_gpu': self.use_gpu,
                'base_output_dir': self.base_output_dir,
                'batch_dir': self.batch_dir
            })
            processed_configs.append(processed_config)
        
        # 使用多进程并行生成
        results = []
        errors = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(process_temperature, config): config 
                for config in processed_configs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成进度"):
                try:
                    result = future.result()
                    if result.get('success', False):
                        results.append(result)
                    else:
                        errors.append(result)
                except Exception as e:
                    errors.append({
                        'error': str(e),
                        'temperature': None,
                        'success': False
                    })
        
        # 计算性能统计
        total_time = time.time() - start_time
        successful_images = sum(r['num_images'] for r in results)
        avg_time_per_image = total_time / successful_images if successful_images > 0 else 0
        generation_rate = successful_images / (total_time / 60) if total_time > 0 else 0
        
        self.performance_stats = {
            'total_images': total_images,
            'successful_images': successful_images,
            'failed_images': total_images - successful_images,
            'total_time': total_time,
            'avg_time_per_image': avg_time_per_image,
            'generation_rate': generation_rate,
            'errors': errors
        }
        
        # 保存批量信息
        self._save_batch_info(processed_configs, results, errors)
        
        # 显示汇总信息
        print("=" * 60)
        print(f"批量生成完成！")
        print(f"总图像数量: {total_images}")
        print(f"成功生成: {successful_images}")
        print(f"生成失败: {total_images - successful_images}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每张图像耗时: {avg_time_per_image:.2f} 秒")
        print(f"生成速率: {generation_rate:.2f} 图像/分钟")
        print(f"输出目录: {self.output_dir}")
        
        if errors:
            print(f"\n警告: 有 {len(errors)} 个温度点处理失败")
        
        return {
            'results': results,
            'performance_stats': self.performance_stats,
            'output_dir': self.output_dir,
            'errors': errors
        }
    
    def _save_batch_info(self, batch_configs, results, errors):
        """保存批量信息"""
        info_path = os.path.join(self.data_dir, "batch_info.txt")
        
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write("XY模型批量自旋图生成信息\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总图像数量: {self.performance_stats['total_images']}\n")
            f.write(f"成功生成: {self.performance_stats['successful_images']}\n")
            f.write(f"生成失败: {self.performance_stats['failed_images']}\n")
            f.write(f"总耗时: {self.performance_stats['total_time']:.2f} 秒\n")
            f.write(f"平均每张图像耗时: {self.performance_stats['avg_time_per_image']:.2f} 秒\n")
            f.write(f"生成速率: {self.performance_stats['generation_rate']:.2f} 图像/分钟\n")
            f.write(f"GPU加速: {'启用' if self.use_gpu else '禁用'}\n")
            f.write(f"并行进程数: {self.num_workers}\n")
            
            if errors:
                f.write(f"\n错误汇总 ({len(errors)}):\n")
                f.write("-" * 30 + "\n")
                for err in errors:
                    f.write(f"温度 {err.get('temperature')}: {err.get('error', '未知错误')}\n")
            
            f.write("\n温度配置:\n")
            f.write("-" * 30 + "\n")
            for config in batch_configs:
                f.write(f"温度: {config['temperature']:.3f}\n")
                f.write(f"  图像数量: {config['num_images']}\n")
                f.write(f"  晶格尺寸: {config['lattice_size']}x{config['lattice_size']}\n")
                f.write(f"  温度范围: {config['temp_range']}\n")
                f.write(f"  温度点数: {config['num_temperatures']}\n")
                f.write(f"  温度类别: {get_temperature_category(config['temperature'])}\n")
            
            f.write("\n输出目录结构:\n")
            f.write("-" * 30 + "\n")
            f.write(f"{self.output_dir}/\n")
            f.write("├── images/\n")
            f.write("│   ├── low_temp/      (低温自旋图)\n")
            f.write("│   ├── medium_temp/   (中温自旋图)\n")
            f.write("│   └── high_temp/     (高温自旋图)\n")
            f.write("└── data/               (数据文件)\n")
    
    def create_image_manifest(self) -> str:
        """创建图像清单文件"""
        manifest_path = os.path.join(self.data_dir, "image_manifest.txt")
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write("图像清单文件\n")
            f.write("=" * 50 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"成功生成: {self.performance_stats['successful_images']}\n")
            f.write("\n文件路径列表:\n")
            f.write("-" * 30 + "\n")
            
            for temp_category in ['low_temp', 'medium_temp', 'high_temp']:
                temp_dir = os.path.join(self.image_dir, temp_category)
                f.write(f"\n{temp_category.upper()} ({temp_dir}):\n")
                
                if os.path.exists(temp_dir):
                    files = sorted(os.listdir(temp_dir))
                    for filename in files:
                        if filename.endswith('.png'):
                            f.write(f"  {filename}\n")
                    f.write(f"  总计: {len(files)} 张图像\n")
        
        return manifest_path


# --------------------------
# 预定义配置
# --------------------------
def create_small_scale_batch() -> List[Dict[str, Any]]:
    """创建小规模批次配置（测试用，~30张图像）"""
    return [
        {
            'temperature': 0.3,
            'num_images': 10,
            'lattice_size': 16,
            'temp_range': (0.25, 0.35),
            'num_temperatures': 3
        },
        {
            'temperature': 1.0,
            'num_images': 10,
            'lattice_size': 16,
            'temp_range': (0.9, 1.1),
            'num_temperatures': 3
        },
        {
            'temperature': 2.0,
            'num_images': 10,
            'lattice_size': 16,
            'temp_range': (1.9, 2.1),
            'num_temperatures': 3
        }
    ]


def create_medium_scale_batch() -> List[Dict[str, Any]]:
    """创建中规模批次配置（研究用，~450张图像）"""
    return [
        {
            'temperature': 0.5,
            'num_images': 100,
            'lattice_size': 32,
            'temp_range': (0.4, 0.6),
            'num_temperatures': 5
        },
        {
            'temperature': 1.0,
            'num_images': 200,
            'lattice_size': 32,
            'temp_range': (0.9, 1.1),
            'num_temperatures': 5
        },
        {
            'temperature': 1.5,
            'num_images': 100,
            'lattice_size': 32,
            'temp_range': (1.4, 1.6),
            'num_temperatures': 5
        },
        {
            'temperature': 2.2,
            'num_images': 50,
            'lattice_size': 32,
            'temp_range': (2.1, 2.3),
            'num_temperatures': 5
        }
    ]


def create_large_scale_batch() -> List[Dict[str, Any]]:
    """创建大规模批次配置（深度学习用，~4700张图像）"""
    return [
        {
            'temperature': 0.3,
            'num_images': 500,
            'lattice_size': 64,
            'temp_range': (0.25, 0.35),
            'num_temperatures': 5
        },
        {
            'temperature': 0.7,
            'num_images': 1000,
            'lattice_size': 64,
            'temp_range': (0.6, 0.8),
            'num_temperatures': 5
        },
        {
            'temperature': 1.0,
            'num_images': 1500,
            'lattice_size': 64,
            'temp_range': (0.9, 1.1),
            'num_temperatures': 5
        },
        {
            'temperature': 1.3,
            'num_images': 1000,
            'lattice_size': 64,
            'temp_range': (1.2, 1.4),
            'num_temperatures': 5
        },
        {
            'temperature': 1.7,
            'num_images': 500,
            'lattice_size': 64,
            'temp_range': (1.6, 1.8),
            'num_temperatures': 5
        },
        {
            'temperature': 2.2,
            'num_images': 200,
            'lattice_size': 64,
            'temp_range': (2.1, 2.3),
            'num_temperatures': 5
        }
    ]


def create_ultra_large_batch() -> List[Dict[str, Any]]:
    """创建超大规模批次配置（高性能计算用，~17000张图像）"""
    return [
        {
            'temperature': 0.2,
            'num_images': 1000,
            'lattice_size': 128,
            'temp_range': (0.15, 0.25),
            'num_temperatures': 5
        },
        {
            'temperature': 0.5,
            'num_images': 2000,
            'lattice_size': 128,
            'temp_range': (0.4, 0.6),
            'num_temperatures': 5
        },
        {
            'temperature': 0.8,
            'num_images': 3000,
            'lattice_size': 128,
            'temp_range': (0.7, 0.9),
            'num_temperatures': 5
        },
        {
            'temperature': 1.0,
            'num_images': 4000,
            'lattice_size': 128,
            'temp_range': (0.9, 1.1),
            'num_temperatures': 5
        },
        {
            'temperature': 1.2,
            'num_images': 3000,
            'lattice_size': 128,
            'temp_range': (1.1, 1.3),
            'num_temperatures': 5
        },
        {
            'temperature': 1.5,
            'num_images': 2000,
            'lattice_size': 128,
            'temp_range': (1.4, 1.6),
            'num_temperatures': 5
        },
        {
            'temperature': 1.8,
            'num_images': 1000,
            'lattice_size': 128,
            'temp_range': (1.7, 1.9),
            'num_temperatures': 5
        },
        {
            'temperature': 2.2,
            'num_images': 500,
            'lattice_size': 128,
            'temp_range': (2.1, 2.3),
            'num_temperatures': 5
        }
    ]


# --------------------------
# 主函数
# --------------------------
def main():
    """主函数：运行批量生成"""
    try:
        # 环境初始化
        if not initialize_environment():
            print("环境初始化失败，程序退出")
            sys.exit(1)
        
        # 选择生成规模
        print("XY模型批量自旋图生成器")
        print("=" * 80)
        print("\n请选择生成规模:")
        print("1. 小规模 (测试用，~30张图像)")
        print("2. 中规模 (研究用，~450张图像)")
        print("3. 大规模 (深度学习用，~4700张图像)")
        print("4. 超大规模 (高性能计算用，~17000张图像)")
        
        choice = input("\n请输入选择 (1-4): ").strip()
        
        if choice == "1":
            batch_configs = create_small_scale_batch()
        elif choice == "2":
            batch_configs = create_medium_scale_batch()
        elif choice == "3":
            batch_configs = create_large_scale_batch()
        elif choice == "4":
            batch_configs = create_ultra_large_batch()
        else:
            print("无效选择，默认运行中规模")
            batch_configs = create_medium_scale_batch()
        
        # 创建生成器并运行
        generator = BatchSpinGenerator(
            base_output_dir="batch_spin_images",
            use_gpu=True,
            num_workers=None
        )
        
        results = generator.generate_batch_parallel(batch_configs)
        
        # 创建图像清单
        manifest_path = generator.create_image_manifest()
        
        print(f"\n图像清单已保存到: {manifest_path}")
        print("批量生成完成！")
        
        # 检查是否有错误
        if results.get('errors'):
            print(f"\n注意: 有 {len(results['errors'])} 个温度点处理失败")
            print("请查看 batch_info.txt 获取详细错误信息")
            sys.exit(1)
        else:
            print("\n所有温度点处理成功！")
            sys.exit(0)
            
    except Exception as e:
        print(f"执行出错: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()