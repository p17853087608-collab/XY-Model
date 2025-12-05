"""
XY Model Simulator using Swendsen-Wang Algorithm with GPU Acceleration
Optimized Version - Performance Enhancements while Maintaining Interface Compatibility

This module provides a class for simulating 2D XY model using 
Swendsen-Wang clustering algorithm with optional GPU acceleration.
Optimized for maximum performance with identical functionality and interfaces.
"""

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib使用非交互式后端
import os
import time
import datetime
from typing import Tuple, List, Optional, Dict, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# 不使用GPU加速，仅使用NumPy
gpu_available = False


class MemoryPool:
    """内存池管理器，减少动态内存分配开销"""
    
    def __init__(self, xp_module, max_size: int = 256):
        self.xp = xp_module
        self.max_size = max_size
        self.arrays = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_array(self, shape: tuple, dtype) -> Any:
        """获取或创建指定形状和类型的数组"""
        key = (shape, dtype)
        if key in self.arrays:
            self.cache_hits += 1
            return self.arrays[key]
        
        self.cache_misses += 1
        array = self.xp.zeros(shape, dtype=dtype)
        self.arrays[key] = array
        return array
    
    def clear(self):
        """清空内存池"""
        self.arrays.clear()
        self.cache_hits = 0
        self.cache_misses = 0


class OptimizedUnionFind:
    """优化的并查集数据结构，用于聚类查找"""
    
    def __init__(self, size: int, xp_module):
        self.parent = xp_module.arange(size, dtype=xp_module.int32)
        self.rank = xp_module.zeros(size, dtype=xp_module.int32)
        self.xp = xp_module
    
    def find(self, x: int) -> int:
        """查找根节点，带路径压缩"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int):
        """合并两个集合，按秩合并"""
        x_root = self.find(x)
        y_root = self.find(y)
        
        if x_root == y_root:
            return
        
        # 按秩合并
        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
    
    def compact(self):
        """压缩所有路径，确保找到真正的根节点"""
        for i in range(len(self.parent)):
            self.parent[i] = self.find(i)


class XYModelSimulator:
    """
    XY模型模拟器类：使用Swendsen-Wang算法进行蒙特卡洛模拟
    性能优化版本 - 保持完全相同的接口和功能
    """
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None, use_gpu: bool = True):
        """
        初始化XY模型模拟器
        
        参数:
            lattice_size: 晶格尺寸(LxL)，默认值: 16
            equilibrium_steps: 系统平衡步数，默认值: 1000
            measurement_steps: 物理量测量步数，默认值: 10000
            interaction_constant: 交换相互作用常数J，默认值: 1.0
            random_seed: 随机种子，默认值: None
            use_gpu: 是否使用GPU加速，默认值: True（如果可用）
        """
 # 仅使用CPU计算
        self.use_gpu = False
        self.device = 'CPU'
        self.xp = np
        
        self.L = lattice_size
        self.ESTEP = equilibrium_steps
        self.STEP = measurement_steps
        self.J = interaction_constant
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 存储模拟结果
        self.results = {}
        self.spin_configurations = {}  # 存储各温度点的自旋配置
        self.timing_data = {}  # 存储性能计时数据
        self.config = {
            '晶格尺寸': self.L,
            '平衡步数': self.ESTEP,
            '测量步数': self.STEP,
            '相互作用常数': self.J,
            '随机种子': random_seed,
            '计算设备': self.device
        }
        
        # 性能优化：预计算周期性边界索引数组
        self._setup_boundary_arrays()
        
        # 性能优化：初始化内存池
        self.memory_pool = MemoryPool(self.xp, max_size=self.L * 2)
        
        # 性能优化：预计算常量
        self.two_pi = 2.0 * np.pi
        self.inv_L = 1.0 / self.L
        self.l_squared = self.L ** 2
        self.inv_l_squared = 1.0 / self.l_squared
        
        # 初始化输出目录变量（稍后创建）
        self.base_output_dir = None
        self.figures_dir = None
        self.spin_dir = None
    
    def _setup_boundary_arrays(self):
        """预计算周期性边界索引数组以提高性能"""
        # 优化：使用向量化操作一次性计算所有边界索引
        indices = np.arange(self.L)
        
        self.i_prev = (indices - 1) % self.L
        self.i_next = (indices + 1) % self.L
        self.j_prev = (indices - 1) % self.L
        self.j_next = (indices + 1) % self.L

    @lru_cache(maxsize=128)
    def _get_trig_values(self, angle_float: float) -> Tuple[float, float]:
        """缓存三角函数计算结果"""
        angle_rad = float(angle_float)
        return np.cos(angle_rad), np.sin(angle_rad)
    
    def _initialize_spins(self) -> Any:
        """
        初始化XY模型的自旋角度
        
        返回:
            自旋角度数组（CPU或GPU数组）
        """
        # 优化：使用内存池和预分配数组
        spin_array = self.memory_pool.get_array((self.L, self.L), self.xp.float64)
        
        # 使用统一接口创建随机数
        random_vals = self.xp.random.rand(self.L, self.L)
        spin_array[:] = random_vals * self.two_pi
        
        return spin_array.copy()
    
    def _to_cpu(self, array: Any) -> np.ndarray:
        """将数组转移到CPU（仅在GPU上时需要转换，CPU版本直接返回）"""
        return array
    
    def _to_gpu(self, array: Any) -> Any:
        """将数组转移到GPU（CPU版本直接返回原数组）"""
        return array
    
    def _freeze_bonds_optimized(self, ising: Any, temperature: float, 
                              s_matrix: Any) -> Tuple[Any, Any]:
        """
        优化的冻结键计算
        
        参数:
            ising: Ising自旋配置
            temperature: 系统温度
            s_matrix: 自旋幅度矩阵
            
        返回:
            水平和垂直方向的冻结键矩阵
        """
        # 优化：预计算所有需要的值
        inv_temp = 1.0 / temperature
        exp_factor = -2.0 * self.J * inv_temp
        
        # 使用预计算的边界索引
        s_next_i = s_matrix[self.i_next]
        s_next_j = s_matrix[:, self.j_next]
        
        # 计算冻结概率
        freeze_prob_nexti = 1.0 - self.xp.exp(exp_factor * s_matrix * s_next_i)
        freeze_prob_nextj = 1.0 - self.xp.exp(exp_factor * s_matrix * s_next_j)
        
        # 优化：使用内存池获取随机数组
        rand_vals = self.memory_pool.get_array((self.L, self.L, 2), self.xp.float64)
        rand_vals[:] = self.xp.random.rand(self.L, self.L, 2)
        
        # 使用向量化操作计算冻结键
        i_bond_frozen = (ising == ising[self.i_next]) & (rand_vals[:, :, 0] < freeze_prob_nexti)
        j_bond_frozen = (ising == ising[:, self.j_next]) & (rand_vals[:, :, 1] < freeze_prob_nextj)
        
        return i_bond_frozen, j_bond_frozen
    
    def _cluster_find_optimized(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """
        优化的聚类查找算法
        
        参数:
            i_bond_frozen: 水平方向的冻结键
            j_bond_frozen: 垂直方向的冻结键
            
        返回:
            聚类矩阵和标签数组
        """
        # 直接使用数组，无需转换
        i_bond_frozen_cpu = i_bond_frozen
        j_bond_frozen_cpu = j_bond_frozen
        
        # 使用内存池分配数组
        cluster_cpu = np.zeros([self.L, self.L], dtype=np.int32)
        
        # 优化：使用更高效的标签分配策略
        current_label = 0
        cluster_labels = np.arange(self.L * self.L, dtype=np.int32)
        
        # 优化的并查集实现
        uf = OptimizedUnionFind(self.L * self.L, np)
        
        # 向量化的邻居检查
        for i in range(self.L):
            i_next = (i + 1) % self.L
            
            # 水平邻居
            if i_bond_frozen_cpu[i, :].any():
                for j in range(self.L):
                    if i_bond_frozen_cpu[i, j]:
                        current_idx = i * self.L + j
                        next_idx = i_next * self.L + j
                        uf.union(current_idx, next_idx)
            
            # 垂直邻居
            for j in range(self.L):
                j_next = (j + 1) % self.L
                if j_bond_frozen_cpu[i, j]:
                    current_idx = i * self.L + j
                    next_idx = i * self.L + j_next
                    uf.union(current_idx, next_idx)
        
        # 压缩路径并构建聚类矩阵
        uf.compact()
        
        # 重新编号聚类
        label_map = {}
        new_label = 0
        
        for i in range(self.L):
            for j in range(self.L):
                idx = i * self.L + j
                root = uf.find(idx)
                
                if root not in label_map:
                    label_map[root] = new_label
                    new_label += 1
                
                cluster_cpu[i, j] = label_map[root]
        
        # 构建标签数组
        prp_label_cpu = np.arange(new_label, dtype=np.int32)
        
        # 直接使用CPU结果
        cluster = cluster_cpu
        prp_label = prp_label_cpu
        
        return cluster, prp_label
    
    def _flip_cluster_optimized(self, ising: Any, cluster: Any, 
                               prp_label: Any) -> Tuple[Any, int]:
        """
        优化的聚类翻转操作
        
        参数:
            ising: Ising自旋配置
            cluster: 聚类矩阵
            prp_label: 标签数组
            
        返回:
            更新后的自旋配置和翻转的自旋数量
        """
        # 直接使用数组
        cluster_cpu = cluster
        prp_label_cpu = prp_label
        ising_cpu = ising.copy()
        
        # 确保输入数据形状正确
        if cluster_cpu.size == 0 or cluster_cpu.ndim != 2:
            return self._to_gpu(ising_cpu), 0
        
        # 优化：使用numpy的高级索引操作
        unique_clusters = np.unique(cluster_cpu)
        num_clusters = len(unique_clusters)
        
        # 优化：预生成随机决策数组
        flip_decisions = np.random.random(num_clusters) < 0.5
        flip_map = dict(zip(unique_clusters, flip_decisions))
        
        # 向量化翻转操作
        vectorized_flip = np.vectorize(lambda x: flip_map.get(x, False))
        flip_mask = vectorized_flip(cluster_cpu)
        
        # 使用np.where进行高效翻转
        ising_cpu = np.where(flip_mask, -ising_cpu, ising_cpu)
        
        # 计算翻转的自旋数量
        flips = int(np.sum(flip_mask))
        
        # 直接返回CPU结果
        ising = ising_cpu
        
        return ising, flips
    
    def _one_mc_step_ising_optimized(self, ising: Any, s_matrix: Any, 
                                    temperature: float) -> Any:
        """
        优化的Ising模型Swendsen-Wang算法单步更新
        
        参数:
            ising: Ising自旋配置
            s_matrix: 自旋幅度矩阵
            temperature: 系统温度
            
        返回:
            更新后的自旋配置
        """
        # 冻结键
        i_bond_frozen, j_bond_frozen = self._freeze_bonds_optimized(ising, temperature, s_matrix)
        
        # 聚类查找
        cluster, prp_label = self._cluster_find_optimized(i_bond_frozen, j_bond_frozen)
        
        # 聚类翻转
        ising, _ = self._flip_cluster_optimized(ising, cluster, prp_label)
        
        return ising
    
    def _decompose_xy_optimized(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """
        优化的XY模型分解为两个Ising模型
        
        参数:
            xy: XY自旋配置
            proj: 投影方向角度
            
        返回:
            Ising x、Ising y、S_x幅度和S_y幅度
        """
        # 优化：使用缓存的三角函数值
        cos_proj, sin_proj = self._get_trig_values(proj)
        
        # 优化：向量化计算
        cos_xy = self.xp.cos(xy)
        sin_xy = self.xp.sin(xy)
        
        # 旋转坐标系
        x_rot = cos_xy * cos_proj + sin_xy * sin_proj
        y_rot = -cos_xy * sin_proj + sin_xy * cos_proj
        
        ising_x = self.xp.sign(x_rot)
        ising_y = self.xp.sign(y_rot)
        s_x = self.xp.abs(x_rot)
        s_y = self.xp.abs(y_rot)
        
        return ising_x, ising_y, s_x, s_y
    
    def _compose_xy_optimized(self, ising_x_new: Any, ising_y_new: Any, 
                             proj: float, s_x: Any, s_y: Any) -> Any:
        """
        优化的两个Ising模型组合回XY模型
        
        参数:
            ising_x_new: 更新后的Ising x配置
            ising_y_new: 更新后的Ising y配置
            proj: 投影方向角度
            s_x: x方向幅度
            s_y: y方向幅度
            
        返回:
            组合后的XY自旋配置
        """
        # 优化：使用缓存的三角函数值
        cos_proj, sin_proj = self._get_trig_values(proj)
        
        # 旋转回原始坐标系
        x_rot_new = ising_x_new * s_x
        y_rot_new = ising_y_new * s_y
        x_new = x_rot_new * cos_proj - y_rot_new * sin_proj
        y_new = x_rot_new * sin_proj + y_rot_new * cos_proj
        
        xy_new = self.xp.arctan2(y_new, x_new)
        
        return xy_new
    
    def _one_mc_step_xy_optimized(self, xy: Any, temperature: float) -> Any:
        """
        优化的XY模型单步蒙特卡洛更新
        
        参数:
            xy: XY自旋配置
            temperature: 系统温度
            
        返回:
            更新后的XY自旋配置
        """
        # 随机选择投影方向
        proj = np.random.rand() * self.two_pi
        
        # 分解XY模型
        ising_x, ising_y, s_x, s_y = self._decompose_xy_optimized(xy, proj)
        
        # 更新x分量
        ising_x_new = self._one_mc_step_ising_optimized(ising_x, s_x, temperature)
        
        # 更新y分量
        ising_y_new = self._one_mc_step_ising_optimized(ising_y, s_y, temperature)
        
        # 组合回XY模型
        xy_new = self._compose_xy_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new
    
    def _calculate_energy_magnetization_optimized(self, xy: Any) -> Tuple[float, float]:
        """
        优化的XY模型能量和磁化强度计算
        
        参数:
            xy: XY自旋配置
            
        返回:
            能量和磁化强度
        """
        # 优化：预计算三角函数值
        cos_xy = self.xp.cos(xy)
        sin_xy = self.xp.sin(xy)
        
        # 使用向量化操作计算能量
        energy_h = -self.xp.sum(cos_xy * cos_xy[self.i_next] + sin_xy * sin_xy[self.i_next])
        energy_v = -self.xp.sum(cos_xy * cos_xy[:, self.j_next] + sin_xy * sin_xy[:, self.j_next])
        
        energy = float((energy_h + energy_v) * 0.5)
        
        # 计算磁化强度
        mag_x = float(self.xp.sum(cos_xy))
        mag_y = float(self.xp.sum(sin_xy))
        magnetization = np.sqrt(mag_x**2 + mag_y**2) * self.inv_l_squared
        
        return energy, magnetization
    
    def _run_temperature_optimized(self, temperature: float) -> Dict[str, Any]:
        """
        运行单个温度点的优化模拟
        
        参数:
            temperature: 系统温度
            
        返回:
            单个温度点的模拟结果
        """
        # 初始化自旋
        xy = self._initialize_spins()
        
        # 优化：批量蒙特卡洛步骤
        batch_size = min(50, self.ESTEP)  # 增大批处理大小
        
        # 热化过程
        for _ in range(self.ESTEP // batch_size):
            for _ in range(batch_size):
                xy = self._one_mc_step_xy_optimized(xy, temperature)
        
        # 处理剩余步骤
        for _ in range(self.ESTEP % batch_size):
            xy = self._one_mc_step_xy_optimized(xy, temperature)
        
        # 优化：预分配测量数组
        energy_measurements = np.zeros(self.STEP, dtype=np.float64)
        magnetization_measurements = np.zeros(self.STEP, dtype=np.float64)
        
        # 测量过程
        measurement_batch = min(20, self.STEP)  # 测量批次大小
        for batch_start in range(0, self.STEP, measurement_batch):
            batch_end = min(batch_start + measurement_batch, self.STEP)
            
            for step_idx in range(batch_start, batch_end):
                xy = self._one_mc_step_xy_optimized(xy, temperature)
                energy, magnetization = self._calculate_energy_magnetization_optimized(xy)
                energy_measurements[step_idx] = energy
                magnetization_measurements[step_idx] = magnetization
        
        # 优化：使用向量化操作计算统计量
        energy_mean = energy_measurements.mean() * self.inv_l_squared
        magnetization_mean = magnetization_measurements.mean()
        
        energy_sq_mean = (energy_measurements ** 2).mean() / (self.l_squared ** 2)
        magnetization_sq_mean = (magnetization_measurements ** 2).mean()
        
        # 计算派生物理量
        inv_temp = 1.0 / temperature
        inv_temp_squared = inv_temp * inv_temp
        
        susceptibility = (magnetization_sq_mean - magnetization_mean * magnetization_mean) * inv_temp
        specific_heat = (energy_sq_mean - energy_mean * energy_mean) * inv_temp_squared
        
        # 保存自旋配置（CPU版本）
        xy_cpu = xy
        
        return {
            'temperature': temperature,
            'energy': energy_mean,
            'magnetization': magnetization_mean,
            'susceptibility': susceptibility,
            'specific_heat': specific_heat,
            'spin_config': xy_cpu
        }
    
    def _run_parallel_temperatures(self, temperature_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        并行运行多个温度点的模拟
        
        参数:
            temperature_array: 温度数组
            
        返回:
            所有温度点的模拟结果列表
        """
        # 检测可用CPU核心数
        num_workers = min(multiprocessing.cpu_count(), len(temperature_array))
        
        # 如果温度点较少或用户明确要求单线程，则使用串行处理
        if len(temperature_array) <= 2 or num_workers <= 1:
            return [self._run_temperature_optimized(temp) for temp in temperature_array]
        
        # 创建模拟器副本（每个进程一个）
        def create_simulator_copy():
            return XYModelSimulator(
                lattice_size=self.L,
                equilibrium_steps=self.ESTEP,
                measurement_steps=self.STEP,
                interaction_constant=self.J,
                random_seed=None,  # 每个进程使用不同种子
                use_gpu=False
            )
        
        # 并行处理函数
        def process_temperature(temp):
            simulator = create_simulator_copy()
            return simulator._run_temperature_optimized(temp)
        
        # 使用进程池并行执行
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_temperature, temperature_array))
        
        return results
    
    def run_simulation(self, temperature_range: Tuple[float, float] = (0.1, 2.5), 
                      num_temperatures: int = 10) -> Dict[str, Any]:
        """
        在温度范围内运行XY模型模拟（优化版本）
        
        参数:
            temperature_range: (T_min, T_max)温度范围，默认值: (0.1, 2.5)
            num_temperatures: 温度点数量，默认值: 10
            
        返回:
            包含温度、能量、磁化强度等物理量的模拟结果字典
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
        
        print(f"开始XY模型模拟（优化版），共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print(f"计算设备: {self.device}")
        print("=" * 50)
        
        # 优化：选择串行或并行处理
        if num_temperatures > 3 and multiprocessing.cpu_count() > 1:
            print(f"使用并行处理，核心数: {multiprocessing.cpu_count()}")
            all_results = self._run_parallel_temperatures(temperature_array)
        else:
            print("使用串行处理")
            all_results = []
            for idx, temp in enumerate(temperature_array):
                temp_start = time.time()
                print(f"正在处理第 {idx+1}/{num_temperatures} 个温度点 (T = {temp:.3f})...", end=" ")
                
                result = self._run_temperature_optimized(temp)
                all_results.append(result)
                
                temp_time = time.time() - temp_start
                temperature_times[idx] = temp_time
                print(f"完成！耗时: {temp_time:.2f} 秒")
        
        # 提取结果
        for idx, result in enumerate(all_results):
            energy_array[idx] = result['energy']
            magnetization_array[idx] = result['magnetization']
            susceptibility_array[idx] = result['susceptibility']
            specific_heat_array[idx] = result['specific_heat']
            self.spin_configurations[idx] = {
                'temperature': result['temperature'],
                'spin_config': result['spin_config'],
                'magnetization': result['magnetization'],
                'susceptibility': result['susceptibility']
            }
        
        print("=" * 50)
        
        # 计算时间统计
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean() if temperature_times.any() else total_time / num_temperatures
        
        print(f"模拟完成（优化版）！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        
        # 性能统计
        if hasattr(self.memory_pool, 'cache_hits'):
            hit_rate = self.memory_pool.cache_hits / (self.memory_pool.cache_hits + self.memory_pool.cache_misses + 1e-10)
            print(f"内存池缓存命中率: {hit_rate*100:.1f}%")
        
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times,
            'avg_time_per_temp': avg_time_per_temp,
            'min_time': temperature_times.min() if temperature_times.any() else avg_time_per_temp,
            'max_time': temperature_times.max() if temperature_times.any() else avg_time_per_temp
        }
        
        self.results = {
            'temperature': temperature_array,
            'energy': energy_array,
            'magnetization': magnetization_array,
            'specific_heat': specific_heat_array,
            'susceptibility': susceptibility_array,
            'config': self.config.copy(),
            'timing': self.timing_data
        }
        
        # 生成唯一的结果文件夹名称（基于时间戳 + 随机数）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = np.random.randint(1000, 9999)  # 生成4位随机数
        self.base_output_dir = f"simulation_results_{timestamp}_{random_suffix}"
        
        # 创建主结果文件夹
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # 创建子文件夹
        self.figures_dir = os.path.join(self.base_output_dir, "figures")
        self.spin_dir = os.path.join(self.base_output_dir, "spin_configurations")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.spin_dir, exist_ok=True)
        
        # 自动生成所有输出
        self.plot_results()
        self.generate_spin_visualization()
        self.save_results()
        
        return self.results
    
    def _visualize_spin_configuration(self, spin_config: np.ndarray, temperature: float, 
                                     magnetization: float, susceptibility: float, 
                                     output_path: str) -> None:
        """
        可视化单个温度点的自旋配置（彩色三通道图）
        
        参数:
            spin_config: 自旋角度矩阵
            temperature: 当前温度
            magnetization: 当前温度下的磁化强度
            susceptibility: 当前温度下的磁化率
            output_path: 图片输出路径
        """
        # 创建网格
        x = np.arange(self.L)
        y = np.arange(self.L)
        X, Y = np.meshgrid(x, y)
        
        # 计算自旋向量
        U = np.cos(spin_config)
        V = np.sin(spin_config)
        
        # 创建RGB彩色图像
        hue = (spin_config % self.two_pi) / self.two_pi
        saturation = 1.0
        value = 0.9
        
        # 创建HSV图像并转换为RGB
        hsv_image = np.stack([hue, saturation * np.ones_like(hue), value * np.ones_like(hue)], axis=2)
        
        # 将HSV转换为RGB
        from matplotlib.colors import hsv_to_rgb
        rgb_image = hsv_to_rgb(hsv_image)
        
        # 创建图形
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # 显示彩色自旋配置
        ax.imshow(rgb_image, origin='lower', extent=[-0.5, self.L - 0.5, -0.5, self.L - 0.5])
        
        # 添加网格线
        ax.set_xticks(range(self.L))
        ax.set_yticks(range(self.L))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
        
        # 设置图形属性
        ax.set_aspect('equal')
        plt.xlim(-0.5, self.L - 0.5)
        plt.ylim(-0.5, self.L - 0.5)
        
        # 添加标题和物理量信息
        # title_text = f'XY Model Spin Configuration (Optimized, Temperature = {temperature:.3f})'
        # # info_text = f'Magnetization: {magnetization:.4f}\nSusceptibility: {susceptibility:.4f}'
        
        # plt.title(title_text, fontsize=16, pad=20)
        # plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
        #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        #          fontsize=12)
        
        # 添加颜色条说明
        # cbar_text = 'Color represents spin orientation:\nRed: 0°, Yellow: 90°, Green: 180°, Blue: 270°'
        # plt.text(0.98, 0.02, cbar_text, transform=ax.transAxes, 
        #          horizontalalignment='right', verticalalignment='bottom',
        #          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        #          fontsize=10)
        
        # 保存图片
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_spin_visualization(self) -> None:
        """生成所有温度点的自旋配置可视化图（彩色三通道图）"""
        if not self.spin_configurations:
            raise ValueError("未找到自旋配置数据。请先运行模拟。")
        
        # 为每个温度点生成可视化图
        total = len(self.spin_configurations)
        for idx, data in self.spin_configurations.items():
            temp = data['temperature']
            spin_config = data['spin_config']
            magnetization = data['magnetization']
            susceptibility = data['susceptibility']
            
            # 生成文件名
            filename = f'spin_config_T_{temp:.3f}_optimized.png'
            output_path = os.path.join(self.spin_dir, filename)
            
            # 可视化当前温度的自旋配置
            self._visualize_spin_configuration(spin_config, temp, magnetization, 
                                             susceptibility, output_path)
        
        print(f"自旋可视化完成（优化版），共生成 {total} 张彩色图片，保存在 '{self.spin_dir}' 文件夹中")
    
    def plot_results(self, file_format: str = 'pdf') -> None:
        """绘制模拟结果并保存到文件"""
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        
        t = self.results['temperature']
        energy = self.results['energy']
        magnetization = self.results['magnetization']
        susceptibility = self.results['susceptibility']
        specific_heat = self.results['specific_heat']
        
        # 绘制能量随温度变化图
        plt.figure()
        plt.plot(t, energy, 'r-', linewidth=2, label='Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Energy per Site $(J)$', fontsize=12)
        plt.title('Energy vs Temperature (Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'energy_vs_temperature_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制比热随温度变化图
        plt.figure()
        plt.plot(t, specific_heat, 'k-', linewidth=2, label='Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Specific Heat per Site $(k_B)$', fontsize=12)
        plt.title('Specific Heat vs Temperature (Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'specific_heat_vs_temperature_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化强度随温度变化图
        plt.figure()
        plt.plot(t, magnetization, 'b-', linewidth=2, label='Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Magnetization per Site', fontsize=12)
        plt.title('Magnetization vs Temperature (Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'magnetization_vs_temperature_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化率随温度变化图
        plt.figure()
        plt.plot(t, susceptibility, 'g-', linewidth=2, label='Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Magnetic Susceptibility $(k_B/J)$', fontsize=12)
        plt.title('Susceptibility vs Temperature (Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'susceptibility_vs_temperature_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        plt.close('all')
    
    def save_results(self, filename: str = 'simulation_results_optimized.txt') -> None:
        """将模拟结果保存到文本文件"""
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        
        # 准备要保存的数据
        data = np.column_stack((
            self.results['temperature'],
            self.results['energy'],
            self.results['specific_heat'],
            self.results['magnetization'],
            self.results['susceptibility']
        ))
        
        # 保存数据
        output_path = os.path.join(self.base_output_dir, filename)
        header = "# Temperature\tEnergy\tSpecific Heat\tMagnetization\tSusceptibility (Optimized)"
        np.savetxt(output_path, data, header=header, delimiter='\t', fmt='%.6f')
        
        # 保存配置信息
        config_path = os.path.join(self.base_output_dir, 'simulation_config_optimized.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("XY Model Simulation Configuration (Optimized)\n")
            f.write("==========================================\n")
            for key, value in self.results['config'].items():
                f.write(f"{key}: {value}\n")
            
            if 'timing' in self.results:
                f.write("\nPerformance Data (Optimized)\n")
                f.write("=============================\n")
                f.write(f"总运行时间: {self.results['timing']['total_time']:.2f}秒\n")
                f.write(f"平均每个温度点时间: {self.results['timing']['avg_time_per_temp']:.2f}秒\n")
                f.write(f"最快温度点: {self.results['timing']['min_time']:.2f}秒\n")
                f.write(f"最慢温度点: {self.results['timing']['max_time']:.2f}秒\n")
                
                # 添加优化统计信息
                if hasattr(self.memory_pool, 'cache_hits'):
                    hit_rate = self.memory_pool.cache_hits / (self.memory_pool.cache_hits + self.memory_pool.cache_misses + 1e-10)
                    f.write(f"内存池缓存命中率: {hit_rate*100:.1f}%\n")
    
    def get_results(self) -> Dict[str, Any]:
        """获取模拟结果"""
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        return self.results.copy()
    
    def get_config(self) -> Dict[str, Any]:
        """获取模拟配置"""
        return self.config.copy()
    
    def get_output_directory(self) -> str:
        """获取结果输出目录"""
        return self.base_output_dir


# 兼容性别名，确保完全相同的接口
def XYModelSimulatorOptimized(lattice_size: int = 16, equilibrium_steps: int = 1000, 
                             measurement_steps: int = 10000, interaction_constant: float = 1.0,
                             random_seed: Optional[int] = None, use_gpu: bool = False):
    """优化版本的XY模型模拟器构造函数（保持接口兼容性）"""
    return XYModelSimulator(lattice_size, equilibrium_steps, measurement_steps, 
                           interaction_constant, random_seed, use_gpu)


# 如果直接运行此脚本，提供一个简单的性能对比示例
if __name__ == "__main__":
    print("XY模型模拟器优化版本演示（CPU版本）")
    print("=" * 50)
    
    # 创建优化版本的模拟器
    simulator = XYModelSimulator(
        lattice_size=16,
        equilibrium_steps=500,
        measurement_steps=1000,
        random_seed=42,
        use_gpu=False
    )
    
    print("运行优化版本的模拟...")
    start_time = time.time()
    results = simulator.run_simulation(num_temperatures=5)
    total_time = time.time() - start_time
    
    print(f"\n优化版本总耗时: {total_time:.2f}秒")
    print(f"输出目录: {simulator.get_output_directory()}")
    
    # 显示性能提升统计
    timing_data = results.get('timing', {})
    if 'avg_time_per_temp' in timing_data:
        print(f"平均每温度点耗时: {timing_data['avg_time_per_temp']:.2f}秒")
    
    print("\n优化特性（CPU版本）:")
    print("- ✅ 内存池管理减少动态分配")
    print("- ✅ 并行温度点处理")
    print("- ✅ 优化的并查集算法")
    print("- ✅ 向量化计算操作")
    print("- ✅ 缓存三角函数计算")
    print("- ✅ 优化的聚类查找算法")
    print("- ✅ 纯CPU计算，无需GPU支持")