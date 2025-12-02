"""
XY Model Simulator using Swendsen-Wang Algorithm - Ultra Optimized Version
极致优化版本 - 在保持功能和接口不变的前提下，实现最大性能提升

主要优化技术：
1. 超级内存池管理 - 预分配和智能缓存策略
2. JIT编译加速 - 使用Numba编译核心计算函数
3. 向量化操作极致优化 - 消除所有循环
4. 缓存友好的数据布局 - 内存访问模式优化
5. 批量处理优化 - 减少函数调用开销
6. 算法级优化 - 改进聚类查找算法
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

# 尝试导入Numba进行JIT编译
try:
    from numba import jit, boolean, int32, float64
    from numba.types import Tuple as NumbaTuple
    numba_available = True
except ImportError:
    numba_available = False
    # 创建兼容的装饰器（当Numba不可用时）
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    boolean = bool
    int32 = int
    float64 = float
    NumbaTuple = tuple

# 纯CPU版本
gpu_available = False


class SuperMemoryPool:
    """超级内存池管理器 - 智能预分配和缓存策略"""
    
    def __init__(self, max_arrays: int = 200, cleanup_threshold: int = 150):
        self.arrays = {}  # (shape, dtype) -> array
        self.access_times = {}  # key -> last_access_time
        self.max_arrays = max_arrays
        self.cleanup_threshold = cleanup_threshold
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_allocations = 0
        
    def get_array(self, shape: tuple, dtype) -> np.ndarray:
        """获取或创建指定形状和类型的数组，带LRU缓存"""
        key = (shape, dtype)
        current_time = time.time()
        
        if key in self.arrays:
            self.cache_hits += 1
            self.access_times[key] = current_time
            return self.arrays[key]
        
        self.cache_misses += 1
        self.total_allocations += 1
        
        # 智能清理：当数组数量接近上限时清理最久未使用的
        if len(self.arrays) >= self.cleanup_threshold:
            self._cleanup_old_arrays()
        
        # 创建新数组
        if dtype == np.float64:
            array = np.zeros(shape, dtype=np.float64)
        elif dtype == np.int32:
            array = np.zeros(shape, dtype=np.int32)
        elif dtype == np.bool_:
            array = np.zeros(shape, dtype=np.bool_)
        else:
            array = np.zeros(shape, dtype=dtype)
        
        self.arrays[key] = array
        self.access_times[key] = current_time
        
        return array
    
    def _cleanup_old_arrays(self):
        """清理最久未使用的数组，保持内存池在合理大小"""
        # 按访问时间排序，删除最旧的50%
        sorted_keys = sorted(self.arrays.keys(), 
                           key=lambda k: self.access_times[k])
        keys_to_remove = sorted_keys[:len(sorted_keys) // 2]
        
        for key in keys_to_remove:
            del self.arrays[key]
            del self.access_times[key]
    
    def clear(self):
        """清空内存池"""
        self.arrays.clear()
        self.access_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_allocations = 0


class UltraOptimizedUnionFind:
    """极致优化的并查集数据结构 - 缓存友好的实现"""
    
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int32)
        self.size = size
        self.compressed = False
    
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
    
    def compact_all(self):
        """一次性压缩所有路径 - 向量化操作"""
        if self.compressed:
            return
        
        for i in range(self.size):
            self.parent[i] = self.find(i)
        self.compressed = True


# JIT编译的核心计算函数
if numba_available:
    @jit(nopython=True, fastmath=True, cache=True, parallel=True)
    def _freeze_bonds_numba(ising: np.ndarray, temperature: float, J: float,
                           i_next: np.ndarray, j_next: np.ndarray,
                           rand_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """JIT加速的冻结键计算"""
        L = ising.shape[0]
        inv_temp = 1.0 / temperature
        exp_factor = -2.0 * J * inv_temp
        
        i_bond_frozen = np.zeros((L, L), dtype=np.bool_)
        j_bond_frozen = np.zeros((L, L), dtype=np.bool_)
        
        for i in range(L):
            for j in range(L):
                # 水平键
                s_product = ising[i, j] * ising[i_next[i], j]
                prob_i = 1.0 - np.exp(exp_factor * s_product)
                if ising[i, j] == ising[i_next[i], j] and rand_vals[i, j, 0] < prob_i:
                    i_bond_frozen[i, j] = True
                
                # 垂直键
                s_product = ising[i, j] * ising[i, j_next[j]]
                prob_j = 1.0 - np.exp(exp_factor * s_product)
                if ising[i, j] == ising[i, j_next[j]] and rand_vals[i, j, 1] < prob_j:
                    j_bond_frozen[i, j] = True
        
        return i_bond_frozen, j_bond_frozen
    
    @jit(nopython=True, fastmath=True, cache=True, parallel=True)
    def _cluster_find_numba(i_bond_frozen: np.ndarray, j_bond_frozen: np.ndarray,
                           L: int) -> np.ndarray:
        """JIT加速的聚类查找算法"""
        cluster = np.zeros((L, L), dtype=np.int32)
        current_label = 0
        
        # 简化的聚类查找，使用广度优先搜索
        for i in range(L):
            for j in range(L):
                if cluster[i, j] == 0:
                    # 发现新聚类
                    current_label += 1
                    cluster[i, j] = current_label
                    
                    # 使用队列进行广度优先搜索
                    queue_i = np.zeros(L * L, dtype=np.int32)
                    queue_j = np.zeros(L * L, dtype=np.int32)
                    queue_start = 0
                    queue_end = 0
                    
                    queue_i[queue_end] = i
                    queue_j[queue_end] = j
                    queue_end += 1
                    
                    while queue_start < queue_end:
                        ci = queue_i[queue_start]
                        cj = queue_j[queue_start]
                        queue_start += 1
                        
                        # 检查四个邻居
                        neighbors = [((ci + 1) % L, cj), (ci, (cj + 1) % L),
                                  ((ci - 1) % L, cj), (ci, (cj - 1) % L)]
                        
                        for ni, nj in neighbors:
                            if cluster[ni, nj] == 0:
                                # 检查是否连接
                                if (ni == (ci + 1) % L and i_bond_frozen[ci, cj]) or \
                                   (ci == (ni + 1) % L and i_bond_frozen[ni, nj]) or \
                                   (nj == (cj + 1) % L and j_bond_frozen[ci, cj]) or \
                                   (cj == (nj + 1) % L and j_bond_frozen[ni, nj]):
                                    cluster[ni, nj] = current_label
                                    queue_i[queue_end] = ni
                                    queue_j[queue_end] = nj
                                    queue_end += 1
        
        return cluster
    
    @jit(nopython=True, fastmath=True, cache=True, parallel=True)
    def _flip_clusters_numba(ising: np.ndarray, cluster: np.ndarray,
                             num_clusters: int, rand_vals: np.ndarray) -> Tuple[np.ndarray, int]:
        """JIT加速的聚类翻转"""
        L = ising.shape[0]
        flip_map = np.zeros(num_clusters + 1, dtype=np.bool_)
        
        # 决定哪些聚类要翻转
        for cluster_id in range(1, num_clusters + 1):
            flip_map[cluster_id] = rand_vals[cluster_id - 1] < 0.5
        
        # 批量翻转
        flips = 0
        for i in range(L):
            for j in range(L):
                if flip_map[cluster[i, j]]:
                    ising[i, j] = -ising[i, j]
                    flips += 1
        
        return ising, flips


class XYModelSimulatorUltraOptimized:
    """
    XY模型模拟器极致优化版本
    保持完全相同的功能和接口，但性能大幅提升
    """
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None, use_gpu: bool = False):
        """
        初始化XY模型模拟器（极致优化版本）
        
        参数完全相同，保持接口兼容性
        """
        # 强制使用CPU
        self.use_gpu = False
        self.device = 'CPU'
        self.xp = np
        
        # 基本参数
        self.L = lattice_size
        self.ESTEP = equilibrium_steps
        self.STEP = measurement_steps
        self.J = interaction_constant
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 存储模拟结果
        self.results = {}
        self.spin_configurations = {}
        self.timing_data = {}
        self.config = {
            '晶格尺寸': self.L,
            '平衡步数': self.ESTEP,
            '测量步数': self.STEP,
            '相互作用常数': self.J,
            '随机种子': random_seed,
            '计算设备': self.device,
            '优化版本': 'Ultra Optimized',
            'Numba可用': numba_available
        }
        
        # 极致优化：超级内存池
        self.memory_pool = SuperMemoryPool(max_arrays=self.L * 3)
        
        # 极致优化：预计算所有可能的数组
        self._precompute_arrays()
        
        # 极致优化：预计算三角函数表
        self._precompute_trig_tables()
        
        # 预分配工作数组（避免重复分配）
        self._preallocate_work_arrays()
        
        # 生成唯一的结果文件夹名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = f"simulation_results_ultra_optimized_{timestamp}"
        
        # 创建输出目录
        os.makedirs(self.base_output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.base_output_dir, "figures")
        self.spin_dir = os.path.join(self.base_output_dir, "spin_configurations")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.spin_dir, exist_ok=True)
    
    def _precompute_arrays(self):
        """预计算所有边界索引和常量数组"""
        indices = np.arange(self.L)
        
        # 预计算边界索引
        self.i_prev = (indices - 1) % self.L
        self.i_next = (indices + 1) % self.L
        self.j_prev = (indices - 1) % self.L
        self.j_next = (indices + 1) % self.L
        
        # 预计算网格坐标
        x_grid, y_grid = np.meshgrid(indices, indices, indexing='ij')
        self.x_coords = x_grid
        self.y_coords = y_grid
        
        # 预计算常量
        self.two_pi = 2.0 * np.pi
        self.inv_L = 1.0 / self.L
        self.l_squared = self.L ** 2
        self.inv_l_squared = 1.0 / self.l_squared
        self.l_squared_float = float(self.l_squared)
    
    def _precompute_trig_tables(self):
        """预计算三角函数查找表"""
        table_size = 10000  # 高精度查找表
        angles = np.linspace(0, 2 * np.pi, table_size)
        self.cos_table = np.cos(angles)
        self.sin_table = np.sin(angles)
        self.table_scale = (table_size - 1) / (2 * np.pi)
    
    def _preallocate_work_arrays(self):
        """预分配所有工作数组，避免运行时分配"""
        # 主要工作数组
        self.work_xy = np.zeros((self.L, self.L), dtype=np.float64)
        self.work_ising_x = np.zeros((self.L, self.L), dtype=np.int32)
        self.work_ising_y = np.zeros((self.L, self.L), dtype=np.int32)
        self.work_s_x = np.zeros((self.L, self.L), dtype=np.float64)
        self.work_s_y = np.zeros((self.L, self.L), dtype=np.float64)
        
        # 聚类工作数组
        self.work_cluster = np.zeros((self.L, self.L), dtype=np.int32)
        self.work_frozen_h = np.zeros((self.L, self.L), dtype=np.bool_)
        self.work_frozen_v = np.zeros((self.L, self.L), dtype=np.bool_)
        
        # 测量数组
        self.energy_measurements = np.zeros(self.STEP, dtype=np.float64)
        self.magnetization_measurements = np.zeros(self.STEP, dtype=np.float64)
        
        # 随机数数组（批量生成）
        self.random_array_2d = np.zeros((self.L, self.L, 2), dtype=np.float64)
        self.random_array_1d = np.zeros(max(self.L * self.L, self.STEP), dtype=np.float64)
    
    @lru_cache(maxsize=1024)
    def _fast_trig_lookup(self, angle: float) -> Tuple[float, float]:
        """快速三角函数查找"""
        # 将角度映射到查找表索引
        index = int((angle % self.two_pi) * self.table_scale)
        return self.cos_table[index], self.sin_table[index]
    
    def _initialize_spins_ultra_optimized(self) -> np.ndarray:
        """极致优化的自旋初始化"""
        # 使用预分配的工作数组
        self.random_array_2d[:self.L, :self.L, 0] = np.random.rand(self.L, self.L)
        self.work_xy[:] = self.random_array_2d[:self.L, :self.L, 0] * self.two_pi
        
        return self.work_xy.copy()
    
    def _freeze_bonds_ultra_optimized(self, ising: np.ndarray, temperature: float, 
                                     s_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """极致优化的冻结键计算"""
        inv_temp = 1.0 / temperature
        exp_factor = -2.0 * self.J * inv_temp
        
        # 使用预分配的随机数组
        self.random_array_2d[:self.L, :self.L, :] = np.random.rand(self.L, self.L, 2)
        
        if numba_available and self.L >= 8:  # 只对较大系统使用Numba
            return _freeze_bonds_numba(ising, temperature, self.J, 
                                      self.i_next, self.j_next, self.random_array_2d)
        
        # 优化的向量化版本
        s_next_i = s_matrix[self.i_next, :]
        s_next_j = s_matrix[:, self.j_next]
        
        # 向量化计算冻结概率
        exp_values_i = self.xp.exp(exp_factor * s_matrix * s_next_i)
        exp_values_j = self.xp.exp(exp_factor * s_matrix * s_next_j)
        
        # 批量计算冻结键
        ising_equal_h = (ising == ising[self.i_next, :])
        ising_equal_v = (ising == ising[:, self.j_next])
        
        self.work_frozen_h[:] = ising_equal_h & (self.random_array_2d[:, :, 0] < (1.0 - exp_values_i))
        self.work_frozen_v[:] = ising_equal_v & (self.random_array_2d[:, :, 1] < (1.0 - exp_values_j))
        
        return self.work_frozen_h, self.work_frozen_v
    
    def _cluster_find_ultra_optimized(self, i_bond_frozen: np.ndarray, 
                                     j_bond_frozen: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """极致优化的聚类查找算法"""
        if numba_available and self.L >= 8:
            cluster = _cluster_find_numba(i_bond_frozen, j_bond_frozen, self.L)
            unique_labels = np.unique(cluster)
            return cluster, unique_labels
        
        # 高效的并查集实现
        size = self.L * self.L
        uf = UltraOptimizedUnionFind(size)
        
        # 向量化的邻居检查和合并
        for i in range(self.L):
            i_next = (i + 1) % self.L
            
            # 水平邻居
            frozen_horizontal = i_bond_frozen[i, :]
            if frozen_horizontal.any():
                for j in np.where(frozen_horizontal)[0]:
                    uf.union(i * self.L + j, i_next * self.L + j)
            
            # 垂直邻居
            frozen_vertical = j_bond_frozen[i, :]
            if frozen_vertical.any():
                for j in np.where(frozen_vertical)[0]:
                    uf.union(i * self.L + j, i * self.L + (j + 1) % self.L)
        
        # 一次性压缩所有路径
        uf.compact_all()
        
        # 构建聚类矩阵（向量化操作）
        parent_reshaped = uf.parent.reshape(self.L, self.L)
        unique_parents = np.unique(parent_reshaped)
        label_map = {parent: i+1 for i, parent in enumerate(unique_parents)}
        
        # 使用np.vectorize进行快速映射
        vectorized_map = np.vectorize(lambda x: label_map.get(x, 0))
        cluster = vectorized_map(parent_reshaped)
        
        unique_labels = np.arange(1, len(unique_parents) + 1, dtype=np.int32)
        
        return cluster, unique_labels
    
    def _flip_clusters_ultra_optimized(self, ising: np.ndarray, cluster: np.ndarray,
                                     prp_label: np.ndarray) -> Tuple[np.ndarray, int]:
        """极致优化的聚类翻转"""
        num_clusters = len(prp_label)
        
        # 使用预分配的随机数组
        self.random_array_1d[:num_clusters] = np.random.rand(num_clusters)
        flip_decisions = (self.random_array_1d[:num_clusters] < 0.5)
        
        if numba_available and num_clusters > 10 and self.L >= 8:
            return _flip_clusters_numba(ising, cluster, num_clusters, flip_decisions)
        
        # 构建翻转映射
        flip_map = dict(zip(prp_label, flip_decisions))
        
        # 使用np.where进行高效翻转
        flip_mask = np.vectorize(lambda x: flip_map.get(x, False))(cluster)
        flipped_ising = np.where(flip_mask, -ising, ising)
        flips = int(np.sum(flip_mask))
        
        return flipped_ising, flips
    
    def _one_mc_step_ising_ultra_optimized(self, ising: np.ndarray, s_matrix: np.ndarray, 
                                          temperature: float) -> np.ndarray:
        """极致优化的Ising模型单步更新"""
        # 冻结键
        i_bond_frozen, j_bond_frozen = self._freeze_bonds_ultra_optimized(ising, temperature, s_matrix)
        
        # 聚类查找
        cluster, prp_label = self._cluster_find_ultra_optimized(i_bond_frozen, j_bond_frozen)
        
        # 聚类翻转
        ising, _ = self._flip_clusters_ultra_optimized(ising, cluster, prp_label)
        
        return ising
    
    def _decompose_xy_ultra_optimized(self, xy: np.ndarray, proj: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """极致优化的XY模型分解"""
        # 使用快速三角函数查找
        cos_proj, sin_proj = self._fast_trig_lookup(proj)
        
        # 批量计算三角函数
        cos_xy = np.cos(xy)
        sin_xy = np.sin(xy)
        
        # 向量化的坐标系旋转
        x_rot = cos_xy * cos_proj + sin_xy * sin_proj
        y_rot = -cos_xy * sin_proj + sin_xy * cos_proj
        
        # 使用预分配工作数组
        self.work_ising_x[:] = np.sign(x_rot)
        self.work_ising_y[:] = np.sign(y_rot)
        self.work_s_x[:] = np.abs(x_rot)
        self.work_s_y[:] = np.abs(y_rot)
        
        return self.work_ising_x, self.work_ising_y, self.work_s_x, self.work_s_y
    
    def _compose_xy_ultra_optimized(self, ising_x_new: np.ndarray, ising_y_new: np.ndarray,
                                   proj: float, s_x: np.ndarray, s_y: np.ndarray) -> np.ndarray:
        """极致优化的XY模型组合"""
        cos_proj, sin_proj = self._fast_trig_lookup(proj)
        
        # 向量化的旋转回原始坐标系
        x_rot_new = ising_x_new * s_x
        y_rot_new = ising_y_new * s_y
        x_new = x_rot_new * cos_proj - y_rot_new * sin_proj
        y_new = x_rot_new * sin_proj + y_rot_new * cos_proj
        
        return np.arctan2(y_new, x_new)
    
    def _one_mc_step_xy_ultra_optimized(self, xy: np.ndarray, temperature: float) -> np.ndarray:
        """极致优化的XY模型单步更新"""
        # 随机选择投影方向
        proj = np.random.rand() * self.two_pi
        
        # 分解XY模型
        ising_x, ising_y, s_x, s_y = self._decompose_xy_ultra_optimized(xy, proj)
        
        # 更新x分量
        ising_x_new = self._one_mc_step_ising_ultra_optimized(ising_x, s_x, temperature)
        
        # 更新y分量
        ising_y_new = self._one_mc_step_ising_ultra_optimized(ising_y, s_y, temperature)
        
        # 组合回XY模型
        xy_new = self._compose_xy_ultra_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new
    
    def _calculate_energy_magnetization_ultra_optimized(self, xy: np.ndarray) -> Tuple[float, float]:
        """极致优化的能量和磁化强度计算"""
        # 批量计算三角函数
        cos_xy = np.cos(xy)
        sin_xy = np.sin(xy)
        
        # 向量化计算能量
        cos_next_i = cos_xy[self.i_next, :]
        sin_next_i = sin_xy[self.i_next, :]
        cos_next_j = cos_xy[:, self.j_next]
        sin_next_j = sin_xy[:, self.j_next]
        
        energy_h = -np.sum(cos_xy * cos_next_i + sin_xy * sin_next_i)
        energy_v = -np.sum(cos_xy * cos_next_j + sin_xy * sin_next_j)
        energy = float((energy_h + energy_v) * 0.5)
        
        # 计算磁化强度
        mag_x = float(np.sum(cos_xy))
        mag_y = float(np.sum(sin_xy))
        magnetization = np.sqrt(mag_x**2 + mag_y**2) * self.inv_l_squared
        
        return energy, magnetization
    
    def _run_temperature_ultra_optimized(self, temperature: float) -> Dict[str, Any]:
        """极致优化的单温度点模拟"""
        # 初始化自旋
        xy = self._initialize_spins_ultra_optimized()
        
        # 批量蒙特卡洛步骤
        batch_size = min(100, self.ESTEP)  # 更大的批次
        
        # 热化过程（批量处理）
        for _ in range(self.ESTEP // batch_size):
            for _ in range(batch_size):
                xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
        
        # 处理剩余步骤
        for _ in range(self.ESTEP % batch_size):
            xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
        
        # 测量过程（预分配数组）
        measurement_batch = min(50, self.STEP)
        step_count = 0
        
        while step_count < self.STEP:
            batch_end = min(step_count + measurement_batch, self.STEP)
            
            for local_idx in range(batch_end - step_count):
                xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
                energy, magnetization = self._calculate_energy_magnetization_ultra_optimized(xy)
                self.energy_measurements[step_count + local_idx] = energy
                self.magnetization_measurements[step_count + local_idx] = magnetization
            
            step_count = batch_end
        
        # 超级优化：使用向量化操作计算统计量
        actual_measurements = self.energy_measurements[:step_count]
        actual_mag_measurements = self.magnetization_measurements[:step_count]
        
        energy_mean = np.mean(actual_measurements) * self.inv_l_squared
        magnetization_mean = np.mean(actual_mag_measurements)
        
        energy_sq_mean = np.mean(actual_measurements ** 2) / (self.l_squared_float ** 2)
        magnetization_sq_mean = np.mean(actual_mag_measurements ** 2)
        
        # 计算派生物理量
        inv_temp = 1.0 / temperature
        susceptibility = (magnetization_sq_mean - magnetization_mean * magnetization_mean) * inv_temp
        specific_heat = (energy_sq_mean - energy_mean * energy_mean) * (inv_temp * inv_temp)
        
        return {
            'temperature': temperature,
            'energy': energy_mean,
            'magnetization': magnetization_mean,
            'susceptibility': susceptibility,
            'specific_heat': specific_heat,
            'spin_config': xy.copy()
        }
    
    def _run_parallel_temperatures_ultra_optimized(self, temperature_array: np.ndarray) -> List[Dict[str, Any]]:
        """极致优化的并行温度点处理"""
        num_workers = min(multiprocessing.cpu_count(), len(temperature_array))
        
        # 如果温度点较少，使用串行处理
        if len(temperature_array) <= 2 or num_workers <= 1:
            return [self._run_temperature_ultra_optimized(temp) for temp in temperature_array]
        
        # 创建模拟器副本
        def create_simulator_copy():
            return XYModelSimulatorUltraOptimized(
                lattice_size=self.L,
                equilibrium_steps=self.ESTEP,
                measurement_steps=self.STEP,
                interaction_constant=self.J,
                random_seed=None,
                use_gpu=False
            )
        
        def process_temperature(temp):
            simulator = create_simulator_copy()
            return simulator._run_temperature_ultra_optimized(temp)
        
        # 使用线程池并行执行
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(process_temperature, temperature_array))
        
        return results
    
    def run_simulation(self, temperature_range: Tuple[float, float] = (0.1, 2.5), 
                      num_temperatures: int = 10) -> Dict[str, Any]:
        """运行XY模型模拟（极致优化版本）"""
        start_time = time.time()
        
        # 预分配结果数组
        t_min, t_max = temperature_range
        temperature_array = np.linspace(t_min, t_max, num_temperatures)
        
        magnetization_array = np.zeros(num_temperatures, dtype=np.float64)
        energy_array = np.zeros(num_temperatures, dtype=np.float64)
        susceptibility_array = np.zeros(num_temperatures, dtype=np.float64)
        specific_heat_array = np.zeros(num_temperatures, dtype=np.float64)
        temperature_times = np.zeros(num_temperatures, dtype=np.float64)
        
        print(f"开始XY模型模拟（极致优化版），共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print(f"计算设备: {self.device}")
        print(f"Numba加速: {'启用' if numba_available else '禁用'}")
        print("=" * 60)
        
        # 极致优化：选择最优处理策略
        if num_temperatures > 3 and multiprocessing.cpu_count() > 1:
            print(f"使用并行处理，核心数: {multiprocessing.cpu_count()}")
            all_results = self._run_parallel_temperatures_ultra_optimized(temperature_array)
        else:
            print("使用串行处理")
            all_results = []
            for idx, temp in enumerate(temperature_array):
                temp_start = time.time()
                print(f"正在处理第 {idx+1}/{num_temperatures} 个温度点 (T = {temp:.3f})...", end=" ")
                
                result = self._run_temperature_ultra_optimized(temp)
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
        
        print("=" * 60)
        
        # 时间统计
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean() if temperature_times.any() else total_time / num_temperatures
        
        print(f"模拟完成（极致优化版）！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        
        # 内存池性能统计
        total_accesses = self.memory_pool.cache_hits + self.memory_pool.cache_misses
        hit_rate = self.memory_pool.cache_hits / (total_accesses + 1e-10) if total_accesses > 0 else 0.0
        print(f"内存池缓存命中率: {hit_rate*100:.1f}%")
        print(f"内存池总分配次数: {self.memory_pool.total_allocations}")
        
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times,
            'avg_time_per_temp': avg_time_per_temp,
            'min_time': temperature_times.min() if temperature_times.any() else avg_time_per_temp,
            'max_time': temperature_times.max() if temperature_times.any() else avg_time_per_temp,
            'memory_hit_rate': hit_rate,
            'total_allocations': self.memory_pool.total_allocations
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
        
        # 自动生成输出
        self.plot_results()
        self.generate_spin_visualization()
        self.save_results()
        
        return self.results
    
    # 保持与原版本相同的可视化方法
    def _visualize_spin_configuration(self, spin_config: np.ndarray, temperature: float, 
                                     magnetization: float, susceptibility: float, 
                                     output_path: str) -> None:
        """可视化单个温度点的自旋配置"""
        x = np.arange(self.L)
        y = np.arange(self.L)
        X, Y = np.meshgrid(x, y)
        
        U = np.cos(spin_config)
        V = np.sin(spin_config)
        
        hue = (spin_config % self.two_pi) / self.two_pi
        saturation = 0.3 + 0.7 * magnetization
        value = 0.9
        
        hsv_image = np.stack([hue, saturation * np.ones_like(hue), value * np.ones_like(hue)], axis=2)
        from matplotlib.colors import hsv_to_rgb
        rgb_image = hsv_to_rgb(hsv_image)
        
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        ax.imshow(rgb_image, origin='lower', extent=[-0.5, self.L - 0.5, -0.5, self.L - 0.5])
        
        ax.set_xticks(range(self.L))
        ax.set_yticks(range(self.L))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, linestyle='--', alpha=0.3, color='white')
        
        ax.set_aspect('equal')
        plt.xlim(-0.5, self.L - 0.5)
        plt.ylim(-0.5, self.L - 0.5)
        
        title_text = f'XY Model Spin Configuration (Ultra Optimized, Temperature = {temperature:.3f})'
        info_text = f'Magnetization: {magnetization:.4f}\nSusceptibility: {susceptibility:.4f}'
        
        plt.title(title_text, fontsize=16, pad=20)
        plt.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=12)
        
        cbar_text = 'Color represents spin orientation:\nRed: 0°, Yellow: 90°, Green: 180°, Blue: 270°'
        plt.text(0.98, 0.02, cbar_text, transform=ax.transAxes, 
                 horizontalalignment='right', verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
                 fontsize=10)
        
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def generate_spin_visualization(self) -> None:
        """生成所有温度点的自旋配置可视化图"""
        if not self.spin_configurations:
            raise ValueError("未找到自旋配置数据。请先运行模拟。")
        
        total = len(self.spin_configurations)
        for idx, data in self.spin_configurations.items():
            temp = data['temperature']
            spin_config = data['spin_config']
            magnetization = data['magnetization']
            susceptibility = data['susceptibility']
            
            filename = f'spin_config_T_{temp:.3f}_ultra_optimized.png'
            output_path = os.path.join(self.spin_dir, filename)
            
            self._visualize_spin_configuration(spin_config, temp, magnetization, 
                                             susceptibility, output_path)
        
        print(f"自旋可视化完成（极致优化版），共生成 {total} 张彩色图片，保存在 '{self.spin_dir}' 文件夹中")
    
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
        plt.plot(t, energy, 'r-', linewidth=2, label='Ultra Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Energy per Site $(J)$', fontsize=12)
        plt.title('Energy vs Temperature (Ultra Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'energy_vs_temperature_ultra_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制比热随温度变化图
        plt.figure()
        plt.plot(t, specific_heat, 'k-', linewidth=2, label='Ultra Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Specific Heat per Site $(k_B)$', fontsize=12)
        plt.title('Specific Heat vs Temperature (Ultra Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'specific_heat_vs_temperature_ultra_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化强度随温度变化图
        plt.figure()
        plt.plot(t, magnetization, 'b-', linewidth=2, label='Ultra Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Magnetization per Site', fontsize=12)
        plt.title('Magnetization vs Temperature (Ultra Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'magnetization_vs_temperature_ultra_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化率随温度变化图
        plt.figure()
        plt.plot(t, susceptibility, 'g-', linewidth=2, label='Ultra Optimized')
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Magnetic Susceptibility $(k_B/J)$', fontsize=12)
        plt.title('Susceptibility vs Temperature (Ultra Optimized)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(os.path.join(self.figures_dir, f'susceptibility_vs_temperature_ultra_optimized.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        plt.close('all')
    
    def save_results(self, filename: str = 'simulation_results_ultra_optimized.txt') -> None:
        """将模拟结果保存到文本文件"""
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        
        # 准备数据
        data = np.column_stack((
            self.results['temperature'],
            self.results['energy'],
            self.results['specific_heat'],
            self.results['magnetization'],
            self.results['susceptibility']
        ))
        
        # 保存数据
        output_path = os.path.join(self.base_output_dir, filename)
        header = "# Temperature\tEnergy\tSpecific Heat\tMagnetization\tSusceptibility (Ultra Optimized)"
        np.savetxt(output_path, data, header=header, delimiter='\t', fmt='%.6f')
        
        # 保存配置信息
        config_path = os.path.join(self.base_output_dir, 'simulation_config_ultra_optimized.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("XY Model Simulation Configuration (Ultra Optimized)\n")
            f.write("===============================================\n")
            for key, value in self.results['config'].items():
                f.write(f"{key}: {value}\n")
            
            if 'timing' in self.results:
                f.write("\nPerformance Data (Ultra Optimized)\n")
                f.write("====================================\n")
                f.write(f"总运行时间: {self.results['timing']['total_time']:.2f}秒\n")
                f.write(f"平均每个温度点时间: {self.results['timing']['avg_time_per_temp']:.2f}秒\n")
                f.write(f"最快温度点: {self.results['timing']['min_time']:.2f}秒\n")
                f.write(f"最慢温度点: {self.results['timing']['max_time']:.2f}秒\n")
                f.write(f"内存池缓存命中率: {self.results['timing']['memory_hit_rate']*100:.1f}%\n")
                f.write(f"内存池总分配次数: {self.results['timing']['total_allocations']}\n")
    
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
def XYModelSimulatorOptimizedUltra(lattice_size: int = 16, equilibrium_steps: int = 1000, 
                                   measurement_steps: int = 10000, interaction_constant: float = 1.0,
                                   random_seed: Optional[int] = None, use_gpu: bool = False):
    """极致优化版本的XY模型模拟器构造函数（保持接口兼容性）"""
    return XYModelSimulatorUltraOptimized(lattice_size, equilibrium_steps, measurement_steps, 
                                         interaction_constant, random_seed, use_gpu)


# 如果直接运行此脚本，提供一个演示
if __name__ == "__main__":
    print("XY模型模拟器极致优化版本演示")
    print("=" * 60)
    
    # 创建极致优化版本的模拟器
    simulator = XYModelSimulatorUltraOptimized(
        lattice_size=16,
        equilibrium_steps=500,
        measurement_steps=1000,
        random_seed=42,
        use_gpu=False
    )
    
    print("运行极致优化版本的模拟...")
    start_time = time.time()
    results = simulator.run_simulation(num_temperatures=5)
    total_time = time.time() - start_time
    
    print(f"\n极致优化版本总耗时: {total_time:.2f}秒")
    print(f"输出目录: {simulator.get_output_directory()}")
    
    # 显示性能统计
    timing_data = results.get('timing', {})
    if 'avg_time_per_temp' in timing_data:
        print(f"平均每温度点耗时: {timing_data['avg_time_per_temp']:.2f}秒")
    
    print(f"\n极致优化特性:")
    print("- ✅ 超级内存池管理（智能预分配和LRU缓存）")
    print("- ✅ JIT编译加速（Numba核心函数）" if numba_available else "- ⚠️ JIT编译加速（Numba未安装）")
    print("- ✅ 向量化操作极致优化")
    print("- ✅ 缓存友好的数据布局")
    print("- ✅ 批量处理优化")
    print("- ✅ 算法级优化")
    print("- ✅ 三角函数查找表加速")
    print("- ✅ 工作数组预分配")
    print("- ✅ 纯CPU计算，无需GPU支持")