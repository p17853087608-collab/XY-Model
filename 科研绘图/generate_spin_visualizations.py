#!/usr/bin/env python3
"""
自旋配置可视化脚本 v3.1 - PDF矢量图版本
只处理直接包含自旋配置文件的文件夹，不处理父文件夹
支持并行PDF矢量图生成，提高处理效率
输出格式为PDF矢量图，适合科研出版
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings

# 禁用matplotlib的多进程警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 确保matplotlib在多进程环境下正常工作
plt.switch_backend('Agg')


def visualize_single_spin(args: Tuple[str, np.ndarray, str, str]) -> Tuple[str, bool, str]:
    """
    单个自旋配置的可视化函数，用于并行处理
    
    参数:
        args: 包含(key, spin_array, output_path, title)的元组
        
    返回:
        (filename, success, error_msg)的元组
    """
    key, spin_array, output_path, title = args
    filename = os.path.basename(output_path)
    
    try:
        # 处理不同维度的自旋配置
        if spin_array.ndim == 2:
            # 2D角度数组
            if spin_array.max() > 2 * np.pi:
                # 可能是度数，转换为弧度
                spin_array = spin_array * np.pi / 180.0
            
            # 映射到[0, 2π]范围
            spin_array = spin_array % (2 * np.pi)
            
            # 转换为灰度图
            grayscale = spin_array / (2 * np.pi)
            
            # 创建纯净图片 - 无边框、无标题、无颜色条
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
            
            # 移除所有冗杂信息
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
        elif spin_array.ndim == 3 and spin_array.shape[2] >= 2:
            # 3D数组，可能是[cos, sin]格式
            if spin_array.shape[2] >= 2:
                cos_xy = spin_array[:, :, 0]
                sin_xy = spin_array[:, :, 1]
                
                # 计算角度
                angles = np.arctan2(sin_xy, cos_xy)
                angles = (angles + 2 * np.pi) % (2 * np.pi)
                
                # 转换为灰度图
                grayscale = angles / (2 * np.pi)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
                
                # 移除所有冗杂信息
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
        else:
            return (filename, False, f"不支持的数组维度: {spin_array.shape}")
        
        # 保存纯净图片（PDF矢量格式）
        plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close()
        return (filename, True, "")
        
    except Exception as e:
        plt.close()
        return (filename, False, str(e))


class SpinConfigurationVisualizer:
    """自旋配置可视化器"""
    
    def __init__(self, enable_parallel: bool = True, max_workers: int = None):
        self.supported_formats = ['.npz', '.npy', '.txt', '.dat', '.json']
        self.processed_folders = []
        self.failed_folders = []
        self.enable_parallel = enable_parallel
        # 默认使用CPU核心数，但最多不超过8个进程（避免内存问题）
        if max_workers is None:
            self.max_workers = min(mp.cpu_count(), 8)
        else:
            self.max_workers = max_workers
        
    def scan_for_spin_data(self, root_dir: str = '.') -> List[str]:
        """
        扫描目录寻找直接包含自旋配置数据的文件夹
        只处理直接包含自旋配置文件的文件夹，不处理父文件夹
        
        参数:
            root_dir: 根目录路径
            
        返回:
            直接包含自旋配置数据的文件夹路径列表
        """
        spin_folders = []
        root_path = Path(root_dir)
        
        print(f"正在扫描目录: {root_path.absolute()}")
        print("=" * 60)
        
        # 首先检查根目录本身是否包含自旋文件
        if root_path.is_dir() and self._has_direct_spin_files(root_path):
            spin_folders.append(str(root_path))
            print(f"✓ 发现直接包含自旋文件的文件夹: {root_path.name}")
        
        # 然后遍历所有子目录
        for folder in root_path.rglob('*'):
            if folder.is_dir() and folder != root_path:
                # 检查文件夹是否直接包含自旋配置数据文件
                if self._has_direct_spin_files(folder):
                    spin_folders.append(str(folder))
                    print(f"✓ 发现直接包含自旋文件的文件夹: {folder.relative_to(root_path)}")
        
        print(f"\n总共找到 {len(spin_folders)} 个直接包含自旋配置的文件夹")
        return spin_folders
    
    def _has_direct_spin_files(self, folder: Path) -> bool:
        """
        检查文件夹是否直接包含自旋配置文件（而不是仅在子文件夹中）
        
        参数:
            folder: 文件夹路径
            
        返回:
            是否直接包含自旋配置文件
        """
        # 检查常见的自旋配置文件名模式
        spin_patterns = [
            'spin', 'spin_config', 'configuration', 'spins',
            'xy_spin', 'model_spin', 'lattice', 'state'
        ]
        
        direct_spin_files = 0
        has_subdirs = False
        
        for file in folder.iterdir():
            if file.is_file():
                filename = file.name.lower()
                
                # 检查文件扩展名
                if any(filename.endswith(ext) for ext in self.supported_formats):
                    # 检查文件名是否包含自旋相关关键词
                    if any(pattern in filename for pattern in spin_patterns):
                        direct_spin_files += 1
                    
                    # 检查特殊文件名（排除这些，它们通常在父文件夹中）
                    if 'raw_simulation_data.npz' in filename or 'simulation_data.npz' in filename:
                        # 这些文件通常在父文件夹中，不算作直接的自旋配置文件
                        pass
                        
            elif file.is_dir():
                has_subdirs = True
        
        # 如果有子文件夹但没有直接的自旋文件，可能是父文件夹
        # 至少要有1个直接的自旋配置文件才处理
        return direct_spin_files > 0
    
    def load_spin_data(self, folder: str) -> Dict[str, Any]:
        """
        从文件夹加载自旋配置数据
        
        参数:
            folder: 文件夹路径
            
        返回:
            包含自旋配置数据的字典
        """
        folder_path = Path(folder)
        spin_data = {}
        
        print(f"  正在读取: {folder_path.name}")
        
        # 尝试不同的文件格式
        for file in folder_path.iterdir():
            if not file.is_file():
                continue
                
            filename = file.name.lower()
            
            # 尝试读取NPY文件
            if filename.endswith('.npy'):
                try:
                    data = np.load(file)
                    spin_data[filename] = data
                    print(f"    ✓ 加载NPY数据: {filename} {data.shape}")
                except Exception as e:
                    print(f"    ✗ 读取NPY文件失败 {file.name}: {e}")
            
            # 尝试读取NPZ文件
            elif filename.endswith('.npz'):
                try:
                    data = np.load(file)
                    # 提取可能的自旋配置数据
                    for key in data.files:
                        if any(pattern in key.lower() for pattern in ['spin', 'config', 'state', 'lattice']):
                            spin_data[key] = data[key]
                            print(f"    ✓ 加载NPZ数据: {key} {data[key].shape}")
                    data.close()
                except Exception as e:
                    print(f"    ✗ 读取NPZ文件失败 {file.name}: {e}")
            
            # 尝试读取TXT文件
            elif filename.endswith('.txt') or filename.endswith('.dat'):
                try:
                    if 'spin' in filename or 'config' in filename:
                        data = np.loadtxt(file)
                        if data.ndim >= 2:
                            spin_data[filename] = data
                            print(f"    ✓ 加载TXT数据: {filename} {data.shape}")
                except Exception as e:
                    print(f"    ✗ 读取TXT文件失败 {file.name}: {e}")
            
            # 尝试读取JSON文件
            elif filename.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        if 'spin_configurations' in json_data:
                            spin_data.update(json_data['spin_configurations'])
                            print(f"    ✓ 加载JSON数据: {filename}")
                except Exception as e:
                    print(f"    ✗ 读取JSON文件失败 {file.name}: {e}")
        
        return spin_data
    
    def visualize_spin_configuration(self, spin_array: np.ndarray, output_path: str, 
                                   title: str = "Spin Configuration") -> bool:
        """
        可视化单个自旋配置数组（纯净版本，无任何冗杂信息）
        
        参数:
            spin_array: 自旋配置数组
            output_path: 输出图片路径
            title: 图片标题（内部使用，不显示）
            
        返回:
            是否成功生成图片
        """
        try:
            # 处理不同维度的自旋配置
            if spin_array.ndim == 2:
                # 2D角度数组
                if spin_array.max() > 2 * np.pi:
                    # 可能是度数，转换为弧度
                    spin_array = spin_array * np.pi / 180.0
                
                # 映射到[0, 2π]范围
                spin_array = spin_array % (2 * np.pi)
                
                # 转换为灰度图
                grayscale = spin_array / (2 * np.pi)
                
                # 创建纯净图片 - 无边框、无标题、无颜色条
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
                
                # 移除所有冗杂信息
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                
            elif spin_array.ndim == 3 and spin_array.shape[2] >= 2:
                # 3D数组，可能是[cos, sin]格式
                if spin_array.shape[2] >= 2:
                    cos_xy = spin_array[:, :, 0]
                    sin_xy = spin_array[:, :, 1]
                    
                    # 计算角度
                    angles = np.arctan2(sin_xy, cos_xy)
                    angles = (angles + 2 * np.pi) % (2 * np.pi)
                    
                    # 转换为灰度图
                    grayscale = angles / (2 * np.pi)
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1)
                    
                    # 移除所有冗杂信息
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    
            else:
                print(f"    ✗ 不支持的数组维度: {spin_array.shape}")
                return False
            
            # 保存纯净图片（PDF矢量格式）
            plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"    ✓ 生成图片: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            print(f"    ✗ 生成图片失败: {e}")
            plt.close()
            return False
    
    def process_folder(self, folder: str) -> bool:
        """
        处理单个文件夹，生成可视化图片
        
        参数:
            folder: 文件夹路径
            
        返回:
            是否成功处理
        """
        try:
            # 加载自旋数据
            spin_data = self.load_spin_data(folder)
            
            if not spin_data:
                print(f"  ✗ 未找到有效的自旋配置数据")
                return False
            
            # 创建输出子文件夹
            output_folder = os.path.join(folder, 'generated_visualizations')
            os.makedirs(output_folder, exist_ok=True)
            
            generated_count = 0
            total_configs = len(spin_data)
            
            print(f"  开始生成 {total_configs} 个自旋配置的图片...")
            
            # 准备并行处理的参数
            vis_tasks = []
            for i, (key, spin_array) in enumerate(spin_data.items()):
                if isinstance(spin_array, np.ndarray):
                    # 生成文件名
                    if isinstance(key, str):
                        clean_key = key.replace('/', '_').replace('\\', '_')
                        filename = f"spin_vis_{clean_key}.pdf"
                    else:
                        filename = f"spin_vis_config_{i:03d}.pdf"
                    
                    output_path = os.path.join(output_folder, filename)
                    title = f"Spin Configuration {i+1}"
                    
                    if isinstance(key, str):
                        title += f" ({key})"
                    
                    # 添加到任务列表
                    vis_tasks.append((key, spin_array, output_path, title))
            
            # 执行并行处理
            if self.enable_parallel and len(vis_tasks) > 1:
                start_time = time.time()
                print(f"  🚀 启动 {self.max_workers} 个进程并行生成 {len(vis_tasks)} 张图片...")
                
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    # 提交所有任务
                    future_to_task = {executor.submit(visualize_single_spin, task): task for task in vis_tasks}
                    
                    # 收集结果
                    for future in as_completed(future_to_task):
                        filename, success, error_msg = future.result()
                        if success:
                            generated_count += 1
                            print(f"    ✓ {filename}")
                        else:
                            print(f"    ✗ {filename}: {error_msg}")
                
                elapsed_time = time.time() - start_time
                print(f"  ⏱️  并行处理耗时: {elapsed_time:.2f} 秒")
                
            else:
                # 串行处理（当禁用并行或只有1个任务时）
                print(f"  📝 串行生成 {len(vis_tasks)} 张图片...")
                for key, spin_array, output_path, title in vis_tasks:
                    if self.visualize_spin_configuration(spin_array, output_path, title):
                        generated_count += 1
            
            # 生成处理报告
            report_path = os.path.join(output_folder, 'visualization_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("Spin Configuration Visualization Report\n")
                f.write("=" * 40 + "\n")
                f.write(f"Source folder: {folder}\n")
                f.write(f"Output folder: {output_folder}\n")
                f.write(f"Total configurations found: {total_configs}\n")
                f.write(f"Successfully generated: {generated_count}\n")
                f.write(f"Failed: {total_configs - generated_count}\n")
                f.write("\nGenerated files:\n")
                for i, key in enumerate(spin_data.keys()):
                    if isinstance(key, str):
                        clean_key = key.replace('/', '_').replace('\\', '_')
                        filename = f"spin_vis_{clean_key}.pdf"
                    else:
                        filename = f"spin_vis_config_{i:03d}.pdf"
                    f.write(f"- {filename}\n")
            
            print(f"  ✓ 成功生成 {generated_count}/{total_configs} 张图片")
            print(f"  ✓ 图片保存在: generated_visualizations/")
            print(f"  ✓ 报告文件: visualization_report.txt")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 处理文件夹失败: {e}")
            return False
    
    def process_all_folders(self, root_dir: str = '.') -> None:
        """
        处理所有包含自旋配置的文件夹
        
        参数:
            root_dir: 根目录路径
        """
        print("🎯 自旋配置可视化脚本 v3.1 - PDF矢量图版本")
        print("=" * 60)
        print("只处理直接包含自旋配置文件的文件夹")
        print("输出格式：PDF矢量图（适合科研出版）")
        if self.enable_parallel:
            print(f"🚀 并行模式已启用，使用 {self.max_workers} 个进程")
        else:
            print("📝 串行模式")
        print("=" * 60)
        
        # 扫描文件夹
        spin_folders = self.scan_for_spin_data(root_dir)
        
        if not spin_folders:
            print("❌ 未找到直接包含自旋配置数据的文件夹")
            return
        
        print(f"\n🚀 开始处理 {len(spin_folders)} 个文件夹...")
        print("=" * 60)
        
        # 处理每个文件夹
        for i, folder in enumerate(spin_folders, 1):
            print(f"\n[{i}/{len(spin_folders)}] 处理文件夹: {Path(folder).name}")
            print("-" * 40)
            
            if self.process_folder(folder):
                self.processed_folders.append(folder)
            else:
                self.failed_folders.append(folder)
        
        # 生成总结报告
        self.generate_summary_report(root_dir)
        
        print("\n" + "=" * 60)
        print("🎉 处理完成！")
        print(f"✅ 成功处理: {len(self.processed_folders)} 个文件夹")
        print(f"❌ 处理失败: {len(self.failed_folders)} 个文件夹")
        
        if self.failed_folders:
            print("\n失败的文件夹:")
            for folder in self.failed_folders:
                print(f"  - {folder}")
    
    def generate_summary_report(self, root_dir: str) -> None:
        """
        生成处理总结报告
        
        参数:
            root_dir: 根目录路径
        """
        report_path = os.path.join(root_dir, 'spin_visualization_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Spin Configuration Visualization Summary Report v3.1 - PDF矢量图版本\n")
            f.write("=" * 50 + "\n")
            f.write(f"Root directory: {os.path.abspath(root_dir)}\n")
            f.write(f"Parallel processing: {'Enabled' if self.enable_parallel else 'Disabled'}\n")
            if self.enable_parallel:
                f.write(f"Max workers: {self.max_workers}\n")
            f.write(f"Total folders processed: {len(self.processed_folders) + len(self.failed_folders)}\n")
            f.write(f"Successfully processed: {len(self.processed_folders)}\n")
            f.write(f"Failed: {len(self.failed_folders)}\n\n")
            
            if self.processed_folders:
                f.write("Successfully processed folders:\n")
                for folder in self.processed_folders:
                    output_folder = os.path.join(folder, 'generated_visualizations')
                    f.write(f"  ✓ {folder} -> {output_folder}\n")
            
            if self.failed_folders:
                f.write("\nFailed folders:\n")
                for folder in self.failed_folders:
                    f.write(f"  ✗ {folder}\n")
        
        print(f"\n📊 总结报告已保存: {report_path}")


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='自旋配置可视化脚本 v3.1 - PDF矢量图版本')
    parser.add_argument('directory', nargs='?', default='.', 
                       help='要处理的目录路径（默认为当前目录）')
    parser.add_argument('--no-parallel', action='store_true', 
                       help='禁用并行处理，使用串行模式')
    parser.add_argument('--workers', type=int, 
                       help='指定并行进程数（默认为CPU核心数，最多8个）')
    
    args = parser.parse_args()
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 确定目标目录
    if args.directory == '.':
        target_dir = script_dir
    else:
        target_dir = args.directory
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"❌ 错误：目录 '{target_dir}' 不存在")
        sys.exit(1)
    
    print(f"📍 目标目录: {target_dir}")
    
    # 创建可视化器并处理
    enable_parallel = not args.no_parallel
    visualizer = SpinConfigurationVisualizer(
        enable_parallel=enable_parallel, 
        max_workers=args.workers
    )
    visualizer.process_all_folders(target_dir)


if __name__ == "__main__":
    main()