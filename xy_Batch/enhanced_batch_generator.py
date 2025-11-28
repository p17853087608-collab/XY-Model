"""
XY模型批量自旋图生成器（增强封装版）

提供高可配置的XY模型自旋图像批量生成功能，支持自定义温度范围、晶格尺寸、图像数量等参数，
通过简洁接口实现复杂批量生成任务。
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# --------------------------
# 核心配置类
# --------------------------
class GeneratorConfig:
    """生成器配置类，集中管理所有可配置参数"""
    def __init__(self):
        # 基本参数
        self.lattice_size = 64               # 晶格尺寸 (LxL)
        self.temp_points = [0.3, 0.7, 1.0, 1.3, 1.7, 2.2]  # 温度点列表
        self.images_per_temp = 100           # 每个温度点图像数量
        self.use_gpu = True                 # 是否使用GPU加速
        self.parallel_workers = None        # 并行进程数（None自动适配）
        self.output_base = "batch_output"    # 基础输出目录
        self.dpi = 150                      # 图像分辨率
        self.equilibrium_steps = 1000       # 平衡步数
        self.measurement_steps = 2000       # 测量步数
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于进程间传递"""
        return {
            "lattice_size": self.lattice_size,
            "temp_points": self.temp_points,
            "images_per_temp": self.images_per_temp,
            "use_gpu": self.use_gpu,
            "parallel_workers": self.parallel_workers,
            "output_base": self.output_base,
            "dpi": self.dpi,
            "equilibrium_steps": self.equilibrium_steps,
            "measurement_steps": self.measurement_steps
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GeneratorConfig':
        """从字典加载配置"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# --------------------------
# 核心生成器类
# --------------------------
class XYBatchGenerator:
    """XY模型批量自旋图生成器主类"""
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        初始化生成器
        
        参数:
            config: 生成器配置对象，None则使用默认配置
        """
        self.config = config or GeneratorConfig()
        self._validate_config()
        self._output_dir = None  # 实际输出目录（运行时生成）
        self._logger = self._init_logger()
        
    def _validate_config(self) -> None:
        """验证配置参数合法性"""
        if self.config.lattice_size < 4 or self.config.lattice_size > 256:
            raise ValueError("晶格尺寸必须在4-256范围内")
        if self.config.images_per_temp < 1:
            raise ValueError("每个温度点图像数量必须≥1")
        if not self.config.temp_points:
            raise ValueError("温度点列表不能为空")
        if any(t <= 0 for t in self.config.temp_points):
            raise ValueError("温度值必须为正数")
    
    def _init_logger(self) -> callable:
        """初始化日志打印函数"""
        def logger(msg: str):
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            print(f"{timestamp} {msg}")
        return logger
    
    def set_output_dir(self, custom_dir: str) -> None:
        """
        设置自定义输出目录
        
        参数:
            custom_dir: 自定义输出路径
        """
        self._output_dir = custom_dir
        self._logger(f"已设置自定义输出目录: {custom_dir}")
    
    def _get_output_dir(self) -> str:
        """获取输出目录（自动生成带时间戳的目录）"""
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            return self._output_dir
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = f"batch_{timestamp}"
        output_dir = os.path.join(self.config.output_base, batch_dir)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _generate_single_temp(self, temp: float) -> Dict[str, Any]:
        """
        生成单个温度点的自旋图像
        
        参数:
            temp: 温度值
            
        返回:
            包含生成结果的字典
        """
        try:
            from xy_model_simulator import XYModelSimulator  # 延迟导入模拟器
            
            # 创建模拟器
            simulator = XYModelSimulator(
                lattice_size=self.config.lattice_size,
                equilibrium_steps=self.config.equilibrium_steps,
                measurement_steps=self.config.measurement_steps,
                use_gpu=self.config.use_gpu,
                random_seed=int(temp * 1000)
            )
            
            # 运行模拟
            temp_range = (temp - 0.05, temp + 0.05)
            results = simulator.run_simulation(
                temperature_range=temp_range,
                num_temperatures=3
            )
            
            # 确定温度类别目录
            temp_category = "low_temp" if temp < 0.5 else \
                           "medium_temp" if temp < 1.5 else "high_temp"
            
            output_dir = os.path.join(self._get_output_dir(), "images", temp_category)
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成图像
            image_paths = []
            for i in range(self.config.images_per_temp):
                # 生成不同随机种子的配置
                sim = XYModelSimulator(
                    lattice_size=self.config.lattice_size,
                    equilibrium_steps=self.config.equilibrium_steps // 2,  # 缩短平衡步数加速
                    measurement_steps=self.config.measurement_steps // 2,
                    use_gpu=self.config.use_gpu,
                    random_seed=int(temp * 1000 + i + 1)
                )
                
                sim_results = sim.run_simulation(
                    temperature_range=(temp - 0.02, temp + 0.02),
                    num_temperatures=1
                )
                
                # 提取自旋配置和物理量
                spin_config = sim_results['spin_configurations'][0]['spin_config']
                magnetization = sim_results['spin_configurations'][0]['magnetization']
                susceptibility = sim_results['spin_configurations'][0]['susceptibility']
                
                # 保存图像
                filename = f"spin_T{temp:.3f}_n{i+1:04d}.png"
                output_path = os.path.join(output_dir, filename)
                
                self._save_spin_image(
                    spin_config=spin_config,
                    temperature=temp,
                    magnetization=magnetization,
                    susceptibility=susceptibility,
                    output_path=output_path
                )
                
                image_paths.append(output_path)
            
            return {
                "temperature": temp,
                "category": temp_category,
                "image_count": len(image_paths),
                "success": True,
                "paths": image_paths
            }
            
        except Exception as e:
            return {
                "temperature": temp,
                "success": False,
                "error": str(e)
            }
    
    def _save_spin_image(self, spin_config: np.ndarray, temperature: float,
                        magnetization: float, susceptibility: float, output_path: str) -> None:
        """保存自旋配置图像"""
        lattice_size = spin_config.shape[0]
        x = np.arange(lattice_size)
        y = np.arange(lattice_size)
        X, Y = np.meshgrid(x, y)
        
        # 计算自旋向量
        U = np.cos(spin_config)
        V = np.sin(spin_config)
        
        # 创建HSV色彩映射
        hue = (spin_config % (2 * np.pi)) / (2 * np.pi)
        saturation = 0.4 + 0.6 * magnetization  # 根据磁化强度调整饱和度
        value = 0.9
        
        # 转换为RGB图像
        from matplotlib.colors import hsv_to_rgb
        hsv_image = np.stack([hue, saturation * np.ones_like(hue), value * np.ones_like(hue)], axis=2)
        rgb_image = hsv_to_rgb(hsv_image)
        
        # 绘制图像
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image, origin='lower', extent=[-0.5, lattice_size-0.5, -0.5, lattice_size-0.5])
        plt.title(f"XY Model Spin Configuration (T={temperature:.3f})")
        plt.xlabel(f"Magnetization: {magnetization:.4f} | Susceptibility: {susceptibility:.4f}")
        plt.colorbar(label="Spin Orientation")
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def generate(self) -> Dict[str, Any]:
        """
        执行批量生成任务
        
        返回:
            包含任务摘要的字典
        """
        start_time = time.time()
        output_dir = self._get_output_dir()
        self._logger(f"开始批量生成任务，输出目录: {output_dir}")
        self._logger(f"配置: 温度点={self.config.temp_points}, 晶格尺寸={self.config.lattice_size}, "
                    f"每温度图像数={self.config.images_per_temp}, GPU={self.config.use_gpu}")
        
        # 准备进程池
        from concurrent.futures import ProcessPoolExecutor
        max_workers = self.config.parallel_workers or (4 if sys.platform.startswith('win') else os.cpu_count())
        results = []
        
        # 并行处理温度点
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._generate_single_temp, temp): temp for temp in self.config.temp_points}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="温度点处理进度"):
                temp = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        self._logger(f"温度点 {temp:.3f} 处理完成，生成 {result['image_count']} 张图像")
                    else:
                        self._logger(f"温度点 {temp:.3f} 处理失败: {result['error']}")
                    results.append(result)
                except Exception as e:
                    self._logger(f"温度点 {temp:.3f} 执行异常: {str(e)}")
                    results.append({"temperature": temp, "success": False, "error": str(e)})
        
        # 生成任务报告
        total_success = sum(r["image_count"] for r in results if r["success"])
        total_failed = sum(self.config.images_per_temp for r in results if not r["success"])
        duration = time.time() - start_time
        
        report = {
            "total_temps": len(self.config.temp_points),
            "success_temps": sum(1 for r in results if r["success"]),
            "failed_temps": sum(1 for r in results if not r["success"]),
            "total_images": total_success + total_failed,
            "success_images": total_success,
            "failed_images": total_failed,
            "duration_seconds": duration,
            "output_dir": output_dir,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存报告
        report_path = os.path.join(output_dir, "batch_report.json")
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self._logger(f"批量生成任务完成！总耗时: {duration:.2f}秒")
        self._logger(f"生成统计: 成功{total_success}张, 失败{total_failed}张, 输出目录: {output_dir}")
        
        return report


# --------------------------
# 命令行执行入口
# --------------------------
def main():
    """命令行模式执行批量生成"""
    import argparse
    
    parser = argparse.ArgumentParser(description="XY模型批量自旋图生成器")
    parser.add_argument("--lattice", type=int, default=64, help="晶格尺寸 (4-256)")
    parser.add_argument("--images", type=int, default=50, help="每个温度点图像数量")
    parser.add_argument("--gpu", action="store_true", help="启用GPU加速")
    parser.add_argument("--output", type=str, help="自定义输出目录")
    parser.add_argument("--temps", type=float, nargs="+", help="自定义温度点列表")
    args = parser.parse_args()
    
    # 创建配置
    config = GeneratorConfig()
    config.lattice_size = args.lattice
    config.images_per_temp = args.images
    config.use_gpu = args.gpu
    if args.temps:
        config.temp_points = args.temps
    
    # 创建生成器并执行
    generator = XYBatchGenerator(config)
    if args.output:
        generator.set_output_dir(args.output)
    
    try:
        generator.generate()
    except Exception as e:
        print(f"执行失败: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()