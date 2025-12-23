"""
XY Model Simulator using Swendsen-Wang Algorithm with Parallel Computing Support
Parallel Optimized Version for HPC Multi-core Platforms

This module provides a class for simulating 2D XY model using 
Swendsen-Wang clustering algorithm with multi-core parallel computing support.
Optimized for maximum performance on HPC platforms with single-node multi-core processors.
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
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import signal
import sys
from contextlib import contextmanager

# 不使用GPU加速，仅使用NumPy
gpu_available = False


@contextmanager
def timeout_context(seconds):
    """超时上下文管理器"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({seconds}秒)")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class UltraSmartMemoryPool:
    """超智能内存池管理器，支持并行环境下的高效内存复用"""
    
    def __init__(self, xp_module, max_arrays_per_shape: int = 20, process_id: int = 0):
        self.xp = xp_module
        self.max_arrays_per_shape = max_arrays_per_shape
        self.pools = {}  # 按形状分类的内存池
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_arrays = 0
        self.process_id = process_id
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # 60秒清理一次
    
    def get_array(self, shape: tuple, dtype) -> Any:
        """获取数组，支持形状近似的复用和定期清理"""
        key = (shape, dtype)
        
        # 定期清理内存池防止内存泄漏
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_expired_arrays()
            self.last_cleanup = current_time
        
        # 精确匹配
        if key in self.pools and self.pools[key]:
            self.cache_hits += 1
            self.total_arrays -= 1
            return self.pools[key].pop()
        
        # 查找更大的数组复用（仅当形状差异不大时）
        for (stored_shape, stored_dtype), arrays in self.pools.items():
            if (stored_dtype == dtype and arrays and 
                len(shape) == len(stored_shape) and
                all(s <= stored_s * 1.1 for s, stored_s in zip(shape, stored_shape))):  # 更严格的匹配
                
                array = arrays.pop()
                self.total_arrays -= 1
                self.cache_hits += 1
                
                # 裁剪到所需大小
                if len(shape) == 1:
                    return array[:shape[0]]
                elif len(shape) == 2:
                    return array[:shape[0], :shape[1]]
                else:
                    return array[tuple(slice(s) for s in shape)]
        
        # 创建新数组
        self.cache_misses += 1
        return self.xp.zeros(shape, dtype=dtype)
    
    def return_array(self, array: Any) -> None:
        """归还数组到内存池"""
        if array is None:
            return
            
        shape, dtype = array.shape, array.dtype
        key = (shape, dtype)
        
        if key not in self.pools:
            self.pools[key] = []
        
        if len(self.pools[key]) < self.max_arrays_per_shape:
            self.pools[key].append(array)
            self.total_arrays += 1
    
    def _cleanup_expired_arrays(self):
        """清理过期的数组以释放内存"""
        for key in list(self.pools.keys()):
            if len(self.pools[key]) > self.max_arrays_per_shape // 2:
                # 保留一半的数组
                self.pools[key] = self.pools[key][:self.max_arrays_per_shape // 2]
                self.total_arrays -= len(self.pools[key]) - (self.max_arrays_per_shape // 2)
    
    def clear(self) -> None:
        """清空所有内存池"""
        self.pools.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_arrays = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取内存池统计信息"""
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses + 1e-10),
            'total_arrays_cached': self.total_arrays,
            'shape_types': len(self.pools),
            'process_id': self.process_id
        }


class AdvancedTrigTable:
    """高级三角函数查找表，支持更高精度和批量优化"""
    
    def __init__(self, resolution: int = 20000):  # 提高分辨率
        self.resolution = resolution
        self.angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        self.cos_table = np.cos(self.angles)
        self.sin_table = np.sin(self.angles)
        self.step = 2*np.pi / resolution
        self.inv_step = 1.0 / self.step
        
        # 预计算常用的平方和平方根值
        self._sincos_cache = {}
    
    def get_cos_sin(self, angle: float) -> Tuple[float, float]:
        """快速查表获取cos和sin值"""
        # 归一化角度到[0, 2π)
        normalized = angle % (2*np.pi)
        # 快速索引计算
        idx = int(normalized * self.inv_step)
        idx = max(0, min(idx, self.resolution - 1))
        return self.cos_table[idx], self.sin_table[idx]
    
    def get_cos_sin_batch(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量查表获取cos和sin值，向量化优化"""
        normalized = angles % (2*np.pi)
        indices = (normalized * self.inv_step).astype(int)
        indices = np.clip(indices, 0, self.resolution - 1)
        return self.cos_table[indices], self.sin_table[indices]
    
    def get_cos_sin_squared_batch(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """批量获取cos, sin, cos², sin²值，用于能量计算优化"""
        cos_vals, sin_vals = self.get_cos_sin_batch(angles)
        return cos_vals, sin_vals, cos_vals * cos_vals, sin_vals * sin_vals


class OptimizedUnionFind:
    """超优化的并查集数据结构，支持并行环境"""
    
    def __init__(self, size: int, xp_module):
        self.parent = xp_module.arange(size, dtype=xp_module.int32)
        self.rank = xp_module.zeros(size, dtype=xp_module.int32)
        self.xp = xp_module
        self.size = size
    
    def find(self, x: int) -> int:
        """查找根节点，使用非递归路径压缩避免栈溢出"""
        root = x
        # 找到根节点
        while self.parent[root] != root:
            root = self.parent[root]
        
        # 路径压缩
        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent
        
        return root
    
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
        for i in range(self.size):
            self.parent[i] = self.find(i)


class MultiIndicatorConvergenceDetector:
    """多指标收敛检测器，提供高精度收敛判断"""
    
    def __init__(self, window_size: int = 20, confidence_level: float = 0.95):
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.energy_history = []
        self.magnetization_history = []
        
    def get_temperature_adaptive_thresholds(self, temperature: float) -> Dict[str, float]:
        """根据温度自适应调整收敛阈值"""
        # 不同温度区域的收敛特性不同
        if temperature < 0.5:  # 低温区：收敛快但涨落小
            energy_threshold = 0.005  # 更严格的能量阈值
            mag_threshold = 0.02      # 磁化强度阈值相对宽松
            trend_threshold = 0.01     # 趋势阈值
        elif temperature < 1.5:  # 临界区：涨落大，需要宽松阈值
            energy_threshold = 0.02   # 能量阈值放宽
            mag_threshold = 0.05     # 磁化强度阈值放宽
            trend_threshold = 0.02   # 趋势阈值放宽
        else:  # 高温区：收敛慢，需要更长时间
            energy_threshold = 0.015  # 中等能量阈值
            mag_threshold = 0.03     # 中等磁化强度阈值
            trend_threshold = 0.015  # 中等趋势阈值
            
        return {
            'energy_relative_std': energy_threshold,
            'magnetization_relative_std': mag_threshold,
            'trend_slope': trend_threshold,
            'autocorrelation': 0.3  # 自相关系数阈值
        }
    
    def calculate_advanced_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """计算高级统计指标"""
        if len(data) < 5:
            return {'mean': 0, 'std': 0, 'trend_slope': 0, 'autocorr': 0}
            
        # 基本统计量
        mean = np.mean(data)
        std = np.std(data)
        
        # 趋势分析（线性回归斜率）
        x = np.arange(len(data))
        if len(x) > 1:
            coeffs = np.polyfit(x, data, 1)
            trend_slope = abs(coeffs[0]) / (abs(mean) + 1e-10)  # 相对斜率
        else:
            trend_slope = 0
        
        # 自相关分析（滞后1）
        if len(data) > 1:
            data_centered = data - mean
            autocorr = np.corrcoef(data_centered[:-1], data_centered[1:])[0, 1]
            autocorr = 0 if np.isnan(autocorr) else abs(autocorr)
        else:
            autocorr = 0
            
        return {
            'mean': mean,
            'std': std,
            'trend_slope': trend_slope,
            'autocorr': autocorr
        }
    
    def check_convergence(self, energy: float, magnetization: float, temperature: float) -> Dict[str, Any]:
        """多指标收敛检测"""
        # 添加新数据点
        self.energy_history.append(energy)
        self.magnetization_history.append(magnetization)
        
        # 保持窗口大小
        if len(self.energy_history) > self.window_size:
            self.energy_history.pop(0)
        if len(self.magnetization_history) > self.window_size:
            self.magnetization_history.pop(0)
        
        # 数据不足，无法判断
        if len(self.energy_history) < 10:
            return {
                'converged': False,
                'confidence': 0.0,
                'reason': '数据点不足',
                'details': {}
            }
        
        # 获取温度自适应阈值
        thresholds = self.get_temperature_adaptive_thresholds(temperature)
        
        # 计算能量统计
        energy_stats = self.calculate_advanced_statistics(np.array(self.energy_history))
        mag_stats = self.calculate_advanced_statistics(np.array(self.magnetization_history))
        
        # 能量收敛判断
        energy_relative_std = energy_stats['std'] / (abs(energy_stats['mean']) + 1e-10)
        energy_converged = (energy_relative_std < thresholds['energy_relative_std'] and
                          energy_stats['trend_slope'] < thresholds['trend_slope'] and
                          energy_stats['autocorr'] < thresholds['autocorrelation'])
        
        # 磁化强度收敛判断
        mag_relative_std = mag_stats['std'] / (abs(mag_stats['mean']) + 1e-10)
        mag_converged = (mag_relative_std < thresholds['magnetization_relative_std'] and
                        mag_stats['trend_slope'] < thresholds['trend_slope'] and
                        mag_stats['autocorr'] < thresholds['autocorrelation'])
        
        # 综合收敛判断（两个指标都收敛才认为收敛）
        converged = energy_converged and mag_converged
        
        # 计算置信度
        confidence = 0.0
        if converged:
            energy_conf = min(1.0, (thresholds['energy_relative_std'] - energy_relative_std) / thresholds['energy_relative_std'])
            mag_conf = min(1.0, (thresholds['magnetization_relative_std'] - mag_relative_std) / thresholds['magnetization_relative_std'])
            confidence = (energy_conf + mag_conf) / 2
        else:
            # 部分收敛也给出部分置信度
            if energy_converged or mag_converged:
                confidence = 0.3
            else:
                confidence = 0.1
        
        details = {
            'energy_relative_std': energy_relative_std,
            'magnetization_relative_std': mag_relative_std,
            'energy_trend': energy_stats['trend_slope'],
            'mag_trend': mag_stats['trend_slope'],
            'energy_autocorr': energy_stats['autocorr'],
            'mag_autocorr': mag_stats['autocorr'],
            'thresholds': thresholds
        }
        
        reason = '多指标收敛' if converged else '未收敛'
        
        return {
            'converged': converged,
            'confidence': confidence,
            'reason': reason,
            'details': details
        }
    
    def reset(self):
        """重置检测器状态"""
        self.energy_history.clear()
        self.magnetization_history.clear()


class ParallelXYModelSimulator:
    """
    并行XY模型模拟器类：使用Swendsen-Wang算法进行蒙特卡洛模拟
    专门针对HPC单节点多核平台优化，支持4核并行计算
    """
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None, use_gpu: bool = False,
                 num_processes: int = 4):
        """
        初始化并行XY模型模拟器
        
        参数:
            lattice_size: 晶格尺寸(LxL)，默认值: 16
            equilibrium_steps: 系统平衡步数，默认值: 1000
            measurement_steps: 物理量测量步数，默认值: 10000
            interaction_constant: 交换相互作用常数J，默认值: 1.0
            random_seed: 随机种子，默认值: None
            use_gpu: 是否使用GPU加速，默认值: False
            num_processes: 并行进程数，默认值: 4
        """
        # 仅使用CPU计算
        self.use_gpu = False
        self.device = 'CPU'
        self.xp = np
        
        self.L = lattice_size
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
        self.spin_configurations = {}
        self.timing_data = {}
        self.config = {
            '晶格尺寸': self.L,
            '平衡步数': self.ESTEP,
            '测量步数': self.STEP,
            '相互作用常数': self.J,
            '随机种子': self.base_seed,
            '计算设备': self.device,
            '并行进程数': self.num_processes
        }
        
        # 性能优化：预计算周期性边界索引数组
        self._setup_boundary_arrays()
        
        # 性能优化：初始化高级内存池
        pool_size = max(15, min(25, self.L // 2))  # 增加内存池容量
        self.memory_pool = UltraSmartMemoryPool(self.xp, max_arrays_per_shape=pool_size, process_id=0)
        
        # 性能优化：预计算常量
        self.two_pi = 2.0 * np.pi
        self.inv_L = 1.0 / self.L
        self.l_squared = self.L ** 2
        self.inv_l_squared = 1.0 / self.l_squared
        
        # 性能优化：初始化高级三角函数查找表
        self.trig_table = AdvancedTrigTable(resolution=20000)
        
        # 初始化输出目录变量
        self.base_output_dir = None
        self.figures_dir = None
        self.spin_dir = None
        
        # 初始化超优化缓存
        self._proj_cache = []
        self._proj_trig_cache = {}
        self._freeze_cache = {}
        self._cluster_cache = {}
        
        # 并行计算优化
        self.temperature_times = None
        self.progress_callback = None
    
    def _setup_boundary_arrays(self):
        """预计算周期性边界索引数组以提高性能"""
        indices = np.arange(self.L)
        
        # 确保索引数组是正确的形状和类型
        self.i_prev = indices - 1
        self.i_next = indices + 1
        self.j_prev = indices - 1
        self.j_next = indices + 1
        
        # 处理边界条件，避免模运算可能的问题
        self.i_prev[self.i_prev < 0] = self.L - 1
        self.i_next[self.i_next >= self.L] = 0
        self.j_prev[self.j_prev < 0] = self.L - 1
        self.j_next[self.j_next >= self.L] = 0

    def _create_process_simulator(self, process_id: int, random_seed: int):
        """为每个进程创建独立的模拟器实例"""
        simulator = ProcessXYModelSimulator(
            lattice_size=self.L,
            equilibrium_steps=self.ESTEP,
            measurement_steps=self.STEP,
            interaction_constant=self.J,
            random_seed=random_seed,
            use_gpu=False,
            process_id=process_id
        )
        return simulator

    @lru_cache(maxsize=256)
    def _get_trig_values(self, angle_float: float) -> Tuple[float, float]:
        """缓存三角函数计算结果，增大缓存"""
        angle_rad = float(angle_float)
        return np.cos(angle_rad), np.sin(angle_rad)
    
    def _initialize_spins(self) -> Any:
        """初始化XY模型的自旋角度"""
        random_vals = self.xp.random.rand(self.L, self.L)
        spin_array = random_vals * self.two_pi
        return spin_array.copy()
    
    def _freeze_bonds_ultra_vectorized(self, ising: Any, temperature: float, 
                                       s_matrix: Any) -> Tuple[Any, Any]:
        """超优化冻结键计算，增强缓存和优化"""
        # 温度相关预计算（缓存常用温度）
        cache_key = (round(temperature, 8), self.L)  # 提高精度
        if cache_key not in self._freeze_cache:
            inv_temp = 1.0 / temperature
            exp_factor = -2.0 * self.J * inv_temp
            
            self._freeze_cache[cache_key] = {
                'exp_factor': exp_factor,
                'inv_temp': inv_temp
            }
        
        cached = self._freeze_cache[cache_key]
        exp_factor = cached['exp_factor']
        
        # 内存池优化分配
        i_bond_frozen = self.memory_pool.get_array((self.L, self.L), np.bool_)
        j_bond_frozen = self.memory_pool.get_array((self.L, self.L), np.bool_)
        
        # 超优化邻居计算
        ising_down = np.roll(ising, -1, axis=0)
        ising_right = np.roll(ising, -1, axis=1)
        s_down = np.roll(s_matrix, -1, axis=0)
        s_right = np.roll(s_matrix, -1, axis=1)
        
        # 批量计算冻结概率，优化分支预测
        product_i = s_matrix * s_down
        product_j = s_matrix * s_right
        
        # 分段优化指数计算
        freeze_prob_i = np.zeros_like(product_i)
        freeze_prob_j = np.zeros_like(product_j)
        
        # 对于小值使用线性近似，对于大值使用精确计算
        mask_small_i = np.abs(product_i) < 0.05
        mask_large_i = ~mask_small_i
        freeze_prob_i[mask_small_i] = -product_i[mask_small_i] * exp_factor
        freeze_prob_i[mask_large_i] = 1.0 - np.exp(exp_factor * product_i[mask_large_i])
        
        mask_small_j = np.abs(product_j) < 0.05
        mask_large_j = ~mask_small_j
        freeze_prob_j[mask_small_j] = -product_j[mask_small_j] * exp_factor
        freeze_prob_j[mask_large_j] = 1.0 - np.exp(exp_factor * product_j[mask_large_j])
        
        # 确保概率在[0,1]范围内
        freeze_prob_i = np.clip(freeze_prob_i, 0.0, 1.0)
        freeze_prob_j = np.clip(freeze_prob_j, 0.0, 1.0)
        
        # 超优化随机决策
        rand_vals = np.random.rand(self.L, self.L, 2)
        
        same_spin_i = (ising == ising_down)
        same_spin_j = (ising == ising_right)
        
        i_bond_frozen[:] = same_spin_i & (rand_vals[:, :, 0] < freeze_prob_i)
        j_bond_frozen[:] = same_spin_j & (rand_vals[:, :, 1] < freeze_prob_j)
        
        # 超早期退出优化
        total_bonds = np.count_nonzero(i_bond_frozen) + np.count_nonzero(j_bond_frozen)
        if total_bonds == 0:
            # 归还数组到内存池
            self.memory_pool.return_array(i_bond_frozen)
            self.memory_pool.return_array(j_bond_frozen)
            
            empty_cluster = np.arange(self.L * self.L, dtype=np.int32).reshape(self.L, self.L)
            empty_labels = np.arange(self.L * self.L, dtype=np.int32)
            return empty_cluster, empty_labels
        
        return i_bond_frozen, j_bond_frozen
    
    def _cluster_find_ultra_optimized(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """超优化聚类查找算法"""
        total_bonds = np.count_nonzero(i_bond_frozen) + np.count_nonzero(j_bond_frozen)
        
        if total_bonds == 0:
            empty_cluster = np.arange(self.L * self.L, dtype=np.int32).reshape(self.L, self.L)
            empty_labels = np.arange(self.L * self.L, dtype=np.int32)
            return empty_cluster, empty_labels
        elif total_bonds < self.L * self.L * 0.05:  # 更稀疏情况
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
        elif self.L <= 64:  # 扩大中等级别晶格范围
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
        else:
            try:
                return self._cluster_find_scipy(i_bond_frozen, j_bond_frozen)
            except ImportError:
                return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
    
    def _cluster_find_scipy(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """使用scipy连通分量算法的高效聚类查找"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        n_nodes = self.L * self.L
        rows, cols = [], []
        
        # 向量化构建连接关系
        h_bonds = np.argwhere(i_bond_frozen)
        if len(h_bonds) > 0:
            i_indices = h_bonds[:, 0]
            j_indices = h_bonds[:, 1]
            
            current_indices = i_indices * self.L + j_indices
            next_i_indices = ((i_indices + 1) % self.L) * self.L + j_indices
            
            rows.extend(current_indices)
            cols.extend(next_i_indices)
            rows.extend(next_i_indices)
            cols.extend(current_indices)
        
        v_bonds = np.argwhere(j_bond_frozen)
        if len(v_bonds) > 0:
            i_indices = v_bonds[:, 0]
            j_indices = v_bonds[:, 1]
            
            current_indices = i_indices * self.L + j_indices
            next_j_indices = i_indices * self.L + ((j_indices + 1) % self.L)
            
            rows.extend(current_indices)
            cols.extend(next_j_indices)
            rows.extend(next_j_indices)
            cols.extend(current_indices)
        
        if len(rows) > 0:
            graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
            n_components, labels = connected_components(graph, directed=False)
        else:
            n_components = n_nodes
            labels = np.arange(n_nodes)
        
        cluster = labels.reshape(self.L, self.L)
        prp_label = np.arange(n_components, dtype=np.int32)
        
        return cluster, prp_label
    
    def _cluster_find_optimized_union_find(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """优化的并查集聚类查找算法"""
        cluster = np.zeros([self.L, self.L], dtype=np.int32)
        uf = OptimizedUnionFind(self.L * self.L, np)
        
        # 批量处理水平连接
        h_bonds = np.argwhere(i_bond_frozen)
        if len(h_bonds) > 0:
            h_current = h_bonds[:, 0] * self.L + h_bonds[:, 1]
            h_next = ((h_bonds[:, 0] + 1) % self.L) * self.L + h_bonds[:, 1]
            
            for current, next_idx in zip(h_current, h_next):
                uf.union(current, next_idx)
        
        # 批量处理垂直连接
        v_bonds = np.argwhere(j_bond_frozen)
        if len(v_bonds) > 0:
            v_current = v_bonds[:, 0] * self.L + v_bonds[:, 1]
            v_next = v_bonds[:, 0] * self.L + ((v_bonds[:, 1] + 1) % self.L)
            
            for current, next_idx in zip(v_current, v_next):
                uf.union(current, next_idx)
        
        uf.compact()
        
        # 向量化重新编号聚类
        all_indices = np.arange(self.L * self.L)
        roots = np.array([uf.find(i) for i in all_indices])
        unique_roots, inverse_indices = np.unique(roots, return_inverse=True)
        
        new_labels = inverse_indices.reshape(self.L, self.L).astype(np.int32)
        cluster = new_labels
        
        if len(unique_roots) == self.L * self.L:
            prp_label = np.arange(len(unique_roots), dtype=np.int32)
        else:
            if len(unique_roots) <= len(inverse_indices):
                final_labels = inverse_indices[:len(unique_roots)]
            else:
                final_labels = np.arange(len(unique_roots))
            prp_label = final_labels.astype(np.int32)
        
        return cluster, prp_label
    
    def _flip_cluster_ultra_optimized(self, ising: Any, cluster: Any, 
                                   prp_label: Any) -> Tuple[Any, int]:
        """超优化聚类翻转操作"""
        unique_clusters = np.unique(cluster)
        num_clusters = len(unique_clusters)
        
        # 超快速决策和映射
        flip_decisions = np.random.random(num_clusters) < 0.5
        flip_map = np.zeros(num_clusters, dtype=np.float32)
        flip_map[flip_decisions] = -2.0
        flip_map[~flip_decisions] = 0.0
        
        # 超快速映射和翻转
        cluster_indices = cluster.astype(np.int32)
        flip_factors = flip_map[cluster_indices] + 1.0
        
        # 就地翻转
        ising = ising * flip_factors
        
        flips = int(np.sum(flip_decisions))
        return ising, flips
    
    def _one_mc_step_ising_ultra_optimized(self, ising: Any, s_matrix: Any, 
                                          temperature: float) -> Any:
        """超优化Ising模型Swendsen-Wang算法单步更新"""
        # 自适应算法选择
        if self.L <= 4:
            return self._one_mc_step_ising_simple(ising, s_matrix, temperature)
        elif temperature < 0.5:
            i_bond_frozen, j_bond_frozen = self._freeze_bonds_ultra_vectorized(ising, temperature, s_matrix)
        else:
            i_bond_frozen, j_bond_frozen = self._freeze_bonds_ultra_vectorized(ising, temperature, s_matrix)
        
        cluster, prp_label = self._cluster_find_ultra_optimized(i_bond_frozen, j_bond_frozen)
        ising, _ = self._flip_cluster_ultra_optimized(ising, cluster, prp_label)
        
        # 归还临时数组到内存池
        self.memory_pool.return_array(i_bond_frozen)
        self.memory_pool.return_array(j_bond_frozen)
        
        return ising
    
    def _one_mc_step_ising_simple(self, ising: Any, s_matrix: Any, temperature: float) -> Any:
        """超简化Ising更新，用于极小晶格"""
        for i in range(self.L):
            for j in range(self.L):
                if np.random.rand() < 0.1:  # 10%概率尝试翻转
                    neighbors_sum = (
                        ising[(i-1)%self.L, j] + ising[(i+1)%self.L, j] +
                        ising[i, (j-1)%self.L] + ising[i, (j+1)%self.L]
                    )
                    delta_E = 2 * self.J * ising[i, j] * neighbors_sum * s_matrix[i, j]
                    
                    if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
                        ising[i, j] *= -1
        return ising
    
    def _decompose_xy_ultra_optimized(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """超优化XY模型分解"""
        proj_key = round(proj, 8)
        if proj_key not in self._proj_trig_cache:
            cos_proj, sin_proj = self.trig_table.get_cos_sin(proj)
            self._proj_trig_cache[proj_key] = (cos_proj, sin_proj)
        
        cos_proj, sin_proj = self._proj_trig_cache[proj_key]
        
        # 批量获取XY角度的三角函数值
        cos_xy, sin_xy = self.trig_table.get_cos_sin_batch(xy)
        
        # 超优化的坐标系旋转
        x_rot = cos_xy * cos_proj + sin_xy * sin_proj
        y_rot = -cos_xy * sin_proj + sin_xy * cos_proj
        
        # 位运算优化符号检测
        x_rot_sign = np.signbit(x_rot)
        y_rot_sign = np.signbit(y_rot)
        
        # 就地符号和幅度计算
        s_x = np.abs(x_rot, out=x_rot)
        s_y = np.abs(y_rot, out=y_rot)
        
        # 使用位运算构造符号数组
        ising_x = np.where(x_rot_sign, -1.0, 1.0).astype(np.float32)
        ising_y = np.where(y_rot_sign, -1.0, 1.0).astype(np.float32)
        
        return ising_x, ising_y, s_x, s_y
    
    def _compose_xy_ultra_optimized(self, ising_x_new: Any, ising_y_new: Any, 
                                   proj: float, s_x: Any, s_y: Any) -> Any:
        """超优化XY模型组合"""
        proj_key = round(proj, 8)
        if proj_key not in self._proj_trig_cache:
            cos_proj, sin_proj = self.trig_table.get_cos_sin(proj)
            self._proj_trig_cache[proj_key] = (cos_proj, sin_proj)
        
        cos_proj, sin_proj = self._proj_trig_cache[proj_key]
        
        # 超优化的坐标系旋转
        x_rot_new = ising_x_new * s_x
        y_rot_new = ising_y_new * s_y
        
        # FMA优化的反旋转
        x_new = x_rot_new * cos_proj - y_rot_new * sin_proj
        y_new = x_rot_new * sin_proj + y_rot_new * cos_proj
        
        # 超优化arctan2计算
        xy_new = np.arctan2(y_new, x_new)
        
        # 确保结果在[0, 2π)范围内
        xy_new = np.where(xy_new < 0, xy_new + self.two_pi, xy_new)
        
        return xy_new
    
    def _one_mc_step_xy_ultra_optimized(self, xy: Any, temperature: float) -> Any:
        """超优化XY模型单步蒙特卡洛更新"""
        # 投影方向缓存优化
        if not hasattr(self, '_proj_cache') or len(self._proj_cache) < 200:
            self._proj_cache = [np.random.rand() * self.two_pi for _ in range(200)]
            self._proj_index = 0
        
        proj = self._proj_cache[self._proj_index % 200]
        self._proj_index += 1
        
        # 超优化的XY分解
        ising_x, ising_y, s_x, s_y = self._decompose_xy_ultra_optimized(xy, proj)
        
        # 并行更新两个分量
        ising_x_new = self._one_mc_step_ising_ultra_optimized(ising_x, s_x, temperature)
        ising_y_new = self._one_mc_step_ising_ultra_optimized(ising_y, s_y, temperature)
        
        # 超优化的XY组合
        xy_new = self._compose_xy_ultra_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new
    
    def _calculate_energy_magnetization_vectorized(self, xy: Any) -> Tuple[float, float]:
        """完全向量化的XY模型能量和磁化强度计算"""
        cos_xy, sin_xy = self.trig_table.get_cos_sin_batch(xy)
        
        # 使用roll实现周期性边界条件
        cos_xy_next_i = np.roll(cos_xy, -1, axis=0)
        cos_xy_next_j = np.roll(cos_xy, -1, axis=1)
        sin_xy_next_i = np.roll(sin_xy, -1, axis=0)
        sin_xy_next_j = np.roll(sin_xy, -1, axis=1)
        
        # 向量化能量计算
        energy_h = -np.sum(cos_xy * cos_xy_next_i + sin_xy * sin_xy_next_i)
        energy_v = -np.sum(cos_xy * cos_xy_next_j + sin_xy * sin_xy_next_j)
        energy = float((energy_h + energy_v) * 0.5)
        
        # 向量化磁化强度计算
        mag_x = float(np.sum(cos_xy))
        mag_y = float(np.sum(sin_xy))
        magnetization = np.sqrt(mag_x**2 + mag_y**2) * self.inv_l_squared
        
        return energy, magnetization
    
    def _run_temperature_ultra_optimized(self, temperature: float) -> Dict[str, Any]:
        """运行单个温度点的超优化模拟，集成多指标收敛检测"""
        try:
            xy = self._initialize_spins()
            
            # === 超优化热化阶段 ===
            # 初始化多指标收敛检测器
            convergence_detector = MultiIndicatorConvergenceDetector(
                window_size=min(30, self.ESTEP // 4), 
                confidence_level=0.95
            )
            
            # 自适应批处理（根据温度优化）
            if temperature < 0.5:  # 低温区收敛快
                if self.ESTEP < 500:
                    batch_size = 2
                    update_freq = 25
                elif self.ESTEP < 2000:
                    batch_size = 8
                    update_freq = 50
                else:
                    batch_size = 15
                    update_freq = 100
            elif temperature < 1.5:  # 临界区需要更精细检测
                if self.ESTEP < 500:
                    batch_size = 1
                    update_freq = 20
                elif self.ESTEP < 2000:
                    batch_size = 3
                    update_freq = 40
                else:
                    batch_size = 5
                    update_freq = 80
            else:  # 高温区
                if self.ESTEP < 500:
                    batch_size = 1
                    update_freq = 30
                elif self.ESTEP < 2000:
                    batch_size = 5
                    update_freq = 60
                else:
                    batch_size = 10
                    update_freq = 120
            
            actual_eq_steps = 0
            consecutive_converged = 0
            high_confidence_count = 0
            min_equilibrium = max(150, self.ESTEP // 3)  # 更严格的最低要求
            convergence_log = []  # 记录收敛过程
            
            for step in range(0, self.ESTEP, batch_size):
                actual_batch = min(batch_size, self.ESTEP - step)
                
                for _ in range(actual_batch):
                    xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
                
                actual_eq_steps += actual_batch
                
                # 增强的多指标收敛检测
                if (actual_eq_steps > self.ESTEP // 3 and  # 更早开始检测
                    actual_eq_steps % update_freq == 0):
                    
                    energy, magnetization = self._calculate_energy_magnetization_vectorized(xy)
                    energy_normalized = energy * self.inv_l_squared
                    
                    # 执行多指标收敛检测
                    convergence_result = convergence_detector.check_convergence(
                        energy_normalized, magnetization, temperature
                    )
                    
                    # 记录收敛过程
                    convergence_log.append({
                        'step': actual_eq_steps,
                        'energy': energy_normalized,
                        'magnetization': magnetization,
                        'converged': convergence_result['converged'],
                        'confidence': convergence_result['confidence'],
                        'reason': convergence_result['reason']
                    })
                    
                    # 高置信度收敛判断
                    if convergence_result['converged']:
                        if convergence_result['confidence'] > 0.8:
                            high_confidence_count += 1
                            if high_confidence_count >= 2:  # 连续2次高置信度
                                remaining_steps = self.ESTEP - actual_eq_steps
                                if remaining_steps > min_equilibrium:
                                    print(f"温度 {temperature:.3f} 在 {actual_eq_steps} 步达到高置信度收敛")
                                    break
                        else:
                            consecutive_converged += 1
                            if consecutive_converged >= 3:  # 连续3次收敛
                                remaining_steps = self.ESTEP - actual_eq_steps
                                if remaining_steps > min_equilibrium:
                                    print(f"温度 {temperature:.3f} 在 {actual_eq_steps} 步达到收敛")
                                    break
                    else:
                        consecutive_converged = 0
                        high_confidence_count = 0
            
            # === 超优化测量阶段 ===
            sampling_interval = max(1, self.STEP // 1000)
            actual_measurements = 0
            
            max_samples = min(1000, self.STEP // sampling_interval)
            energy_samples = np.zeros(max_samples, dtype=np.float32)
            mag_samples = np.zeros(max_samples, dtype=np.float32)
            
            measurement_batch = max(1, min(50, self.STEP // 100))
            
            for batch_start in range(0, self.STEP, measurement_batch):
                batch_end = min(batch_start + measurement_batch, self.STEP)
                
                for step_idx in range(batch_start, batch_end):
                    xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
                    
                    if actual_measurements < max_samples and step_idx % sampling_interval == 0:
                        energy, magnetization = self._calculate_energy_magnetization_vectorized(xy)
                        
                        if np.isfinite(energy) and np.isfinite(magnetization):
                            energy_samples[actual_measurements] = energy * self.inv_l_squared
                            mag_samples[actual_measurements] = magnetization
                            actual_measurements += 1
                
                # 增强的垃圾回收
                if batch_start % (measurement_batch * 5) == 0:
                    import gc
                    gc.collect()
            
            # === 超优化统计分析 ===
            valid_energy = energy_samples[:actual_measurements]
            valid_magnetization = mag_samples[:actual_measurements]
            
            if len(valid_energy) == 0:
                energy_mean = 0.0
                magnetization_mean = 0.0
                susceptibility = 0.0
                specific_heat = 0.0
            else:
                energy_mean = float(np.mean(valid_energy))
                magnetization_mean = float(np.mean(valid_magnetization))
                
                if len(valid_energy) >= 10:
                    energy_sq_mean = float(np.mean(valid_energy * valid_energy))
                    mag_sq_mean = float(np.mean(valid_magnetization * valid_magnetization))
                else:
                    energy_sq_mean = energy_mean ** 2
                    mag_sq_mean = magnetization_mean ** 2
                
                inv_temp = 1.0 / temperature
                inv_temp_squared = inv_temp * inv_temp
                
                susceptibility = max(0.0, (mag_sq_mean - magnetization_mean * magnetization_mean) * inv_temp * self.l_squared)
                specific_heat = max(0.0, (energy_sq_mean - energy_mean * energy_mean) * inv_temp_squared * self.l_squared)
            
            if xy is None or not isinstance(xy, np.ndarray):
                xy = self._initialize_spins()
            
            # 内存优化清理
            del valid_energy, valid_magnetization
            if 'convergence_detector' in locals():
                convergence_detector.reset()
            import gc
            gc.collect()
            
            xy_cpu = xy.copy() if hasattr(xy, 'copy') else np.array(xy)
            
            # 提取收敛信息
            final_convergence_info = {}
            if len(convergence_log) > 0:
                final_log = convergence_log[-1]
                final_convergence_info = {
                    'converged': final_log['converged'],
                    'confidence': final_log['confidence'],
                    'reason': final_log['reason'],
                    'detection_steps': len(convergence_log)
                }
            else:
                final_convergence_info = {
                    'converged': False,
                    'confidence': 0.0,
                    'reason': '未进行收敛检测',
                    'detection_steps': 0
                }
            
            return {
                'temperature': temperature,
                'energy': energy_mean,
                'magnetization': magnetization_mean,
                'susceptibility': susceptibility,
                'specific_heat': specific_heat,
                'spin_config': xy_cpu,
                'actual_equilibrium_steps': actual_eq_steps,
                'actual_measurements': actual_measurements,
                'convergence_info': final_convergence_info,
                'convergence_log': convergence_log[-5:] if len(convergence_log) > 5 else convergence_log
            }
            
        except Exception as e:
            print(f"运行温度 {temperature} 时出错: {e}")
            xy_safe = self._initialize_spins()
            return {
                'temperature': temperature,
                'energy': 0.0,
                'magnetization': 0.0,
                'susceptibility': 0.0,
                'specific_heat': 0.0,
                'spin_config': xy_safe,
                'actual_equilibrium_steps': 0,
                'actual_measurements': 0
            }


class ProcessXYModelSimulator(ParallelXYModelSimulator):
    """单进程XY模型模拟器，用于并行计算"""
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None, use_gpu: bool = False,
                 process_id: int = 0):
        """
        初始化进程专用模拟器
        
        参数:
            process_id: 进程ID，用于区分不同进程
        """
        super().__init__(lattice_size, equilibrium_steps, measurement_steps, 
                        interaction_constant, random_seed, use_gpu, 1)  # 单进程模式
        
        self.process_id = process_id
        # 为每个进程创建独立的内存池
        self.memory_pool = UltraSmartMemoryPool(self.xp, max_arrays_per_shape=20, process_id=process_id)
        
        # 设置进程独立的随机种子
        if random_seed is not None:
            process_seed = random_seed + process_id * 10000
            np.random.seed(process_seed)
            self.process_seed = process_seed
        else:
            self.process_seed = None


def run_single_temperature_parallel(args):
    """
    并行运行单个温度点的函数
    
    参数:
        args: (idx, temperature, config_dict, process_id)
        
    返回:
        (idx, result)
    """
    idx, temperature, config_dict, process_id = args
    
    try:
        # 为每个进程创建独立的模拟器实例
        simulator = ProcessXYModelSimulator(
            lattice_size=config_dict['lattice_size'],
            equilibrium_steps=config_dict['equilibrium_steps'],
            measurement_steps=config_dict['measurement_steps'],
            interaction_constant=config_dict['interaction_constant'],
            random_seed=config_dict['random_seed'],
            use_gpu=config_dict.get('use_gpu', False),
            process_id=process_id
        )
        
        # 运行单个温度点
        result = simulator._run_temperature_ultra_optimized(temperature)
        result['original_index'] = idx
        result['process_id'] = process_id
        
        return idx, result, None  # idx, result, error
        
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
            'convergence_info': {
                'converged': False,
                'confidence': 0.0,
                'reason': '处理错误',
                'detection_steps': 0
            },
            'convergence_log': [],
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


class ParallelXYModelSimulator(ParallelXYModelSimulator):
    """增强的并行XY模型模拟器，支持HPC多核优化"""
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: Optional[int] = None, use_gpu: bool = False,
                 num_processes: int = 4):
        super().__init__(lattice_size, equilibrium_steps, measurement_steps, 
                        interaction_constant, random_seed, use_gpu, num_processes)
    
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
        self.figures_dir = os.path.join(self.base_output_dir, "figures")
        self.spin_dir = os.path.join(self.base_output_dir, "spin_configurations")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.spin_dir, exist_ok=True)
        
        print(f"开始并行XY模型模拟，共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print(f"并行进程数: {self.num_processes}")
        print(f"计算设备: {self.device}")
        print(f"结果保存目录: {self.base_output_dir}")
        print("=" * 60)
        
        # 准备并行任务
        config_dict = {
            'lattice_size': self.L,
            'equilibrium_steps': self.ESTEP,
            'measurement_steps': self.STEP,
            'interaction_constant': self.J,
            'random_seed': self.base_seed,
            'use_gpu': self.use_gpu
        }
        
        tasks = [(idx, temp, config_dict, idx % self.num_processes) 
                for idx, temp in enumerate(temperature_array)]
        
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
                        
                        # 显示收敛信息
                        convergence_info = result.get('convergence_info', {})
                        if convergence_info.get('converged', False):
                            print(f"完成温度 {result['temperature']:.3f} - 收敛成功 (置信度: {convergence_info.get('confidence', 0):.2f})")
                        else:
                            print(f"完成温度 {result['temperature']:.3f} - 未收敛 (进度: {len(completed_indices)}/{num_temperatures})")
                    
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
                # 转换numpy类型为Python原生类型
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
                
                # 收敛检测分析
                if 'detailed_results' in self.results and len(self.results['detailed_results']) > 0:
                    detailed_results = self.results['detailed_results']
                    converged_count = sum(1 for r in detailed_results if r.get('convergence_info', {}).get('converged', False))
                    total_count = len(detailed_results)
                    convergence_rate = converged_count / total_count * 100
                    
                    avg_confidence = 0.0
                    if converged_count > 0:
                        confidences = [r.get('convergence_info', {}).get('confidence', 0) 
                                     for r in detailed_results if r.get('convergence_info', {}).get('converged', False)]
                        avg_confidence = np.mean(confidences) if confidences else 0.0
                    
                    f.write(f"\n智能收敛检测分析:\n")
                    f.write(f"  总温度点数: {total_count}\n")
                    f.write(f"  成功收敛点数: {converged_count}\n")
                    f.write(f"  收敛成功率: {convergence_rate:.1f}%\n")
                    f.write(f"  平均收敛置信度: {avg_confidence:.3f}\n")
                    
                    # 按温度区域分析收敛情况
                    low_temp_results = [r for r in detailed_results if r['temperature'] < 0.5]
                    critical_temp_results = [r for r in detailed_results if 0.5 <= r['temperature'] < 1.5]
                    high_temp_results = [r for r in detailed_results if r['temperature'] >= 1.5]
                    
                    def analyze_convergence_by_region(results, region_name):
                        if len(results) == 0:
                            return
                        converged = sum(1 for r in results if r.get('convergence_info', {}).get('converged', False))
                        rate = converged / len(results) * 100
                        if converged > 0:
                            confidences = [r.get('convergence_info', {}).get('confidence', 0) 
                                         for r in results if r.get('convergence_info', {}).get('converged', False)]
                            avg_conf = np.mean(confidences) if confidences else 0.0
                            f.write(f"  {region_name}: {converged}/{len(results)} ({rate:.1f}%), 平均置信度: {avg_conf:.3f}\n")
                        else:
                            f.write(f"  {region_name}: {converged}/{len(results)} ({rate:.1f}%)\n")
                    
                    analyze_convergence_by_region(low_temp_results, "低温区(T<0.5)")
                    analyze_convergence_by_region(critical_temp_results, "临界区(0.5≤T<1.5)")
                    analyze_convergence_by_region(high_temp_results, "高温区(T≥1.5)")
                
                f.write("\n")
                
                # 临界温度估计
                tc_sus = temps[sus.argmax()]
                tc_heat = temps[heat.argmax()]
                tc_estimate = (tc_sus + tc_heat) / 2
                
                f.write("临界温度分析:\n")
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
            
            # 尝试生成图表
            self._generate_plots()
            
        except Exception as e:
            print(f"生成分析报告时出错: {e}")
    
    def _generate_plots(self):
        """生成结果图表"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置后端避免显示问题
            matplotlib.use('Agg')
            
            # 创建图表目录
            figures_dir = os.path.join(self.base_output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # 设置英文字体，避免中文字体问题
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            temps = self.results['temperature']
            energy = self.results['energy']
            mag = self.results['magnetization']
            sus = self.results['susceptibility']
            heat = self.results['specific_heat']
            
            # 创建综合图表
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'XY模型并行模拟结果 (L={self.lattice_size}, {self.num_processes}进程)', fontsize=14)
            
            # 能量 vs 温度
            ax1.plot(temps, energy, 'b-', linewidth=2, markersize=4)
            ax1.set_xlabel('温度 (T)')
            ax1.set_ylabel('能量 (E)')
            ax1.set_title('能量 vs 温度')
            ax1.grid(True, alpha=0.3)
            
            # 磁化强度 vs 温度
            ax2.plot(temps, mag, 'r-', linewidth=2, markersize=4)
            ax2.set_xlabel('温度 (T)')
            ax2.set_ylabel('磁化强度 (M)')
            ax2.set_title('磁化强度 vs 温度')
            ax2.grid(True, alpha=0.3)
            
            # 磁化率 vs 温度
            ax3.plot(temps, sus, 'g-', linewidth=2, markersize=4)
            ax3.set_xlabel('温度 (T)')
            ax3.set_ylabel('磁化率 (χ)')
            ax3.set_title('磁化率 vs 温度')
            ax3.grid(True, alpha=0.3)
            
            # 比热 vs 温度
            ax4.plot(temps, heat, 'm-', linewidth=2, markersize=4)
            ax4.set_xlabel('温度 (T)')
            ax4.set_ylabel('比热 (C)')
            ax4.set_title('比热 vs 温度')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存综合图表
            plot_file = os.path.join(figures_dir, "xy_model_comprehensive.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建性能分析图表
            fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
            fig2.suptitle('并行计算性能分析', fontsize=14)
            
            # 温度点耗时分布
            temp_times = self.timing_data['per_temperature_time']
            if len(temp_times) > 0:
                ax5.bar(range(len(temp_times)), temp_times, alpha=0.7, color='skyblue')
                ax5.set_xlabel('温度点索引')
                ax5.set_ylabel('耗时 (秒)')
                ax5.set_title('各温度点计算耗时')
                ax5.grid(True, alpha=0.3)
                
                # 添加平均线
                avg_time = self.timing_data['avg_time_per_temp']
                ax5.axhline(y=avg_time, color='red', linestyle='--', label=f'平均 {avg_time:.1f}s')
                ax5.legend()
            
            # 并行效率分析
            categories = ['理论最大', '实际达到']
            efficiencies = [100.0, min(self.timing_data['parallel_efficiency'], 150)]  # 限制显示范围
            colors = ['lightcoral', 'lightgreen']
            
            bars = ax6.bar(categories, efficiencies, color=colors, alpha=0.7)
            ax6.set_ylabel('并行效率 (%)')
            ax6.set_title(f'并行效率对比 (进程数: {self.num_processes})')
            ax6.set_ylim(0, max(efficiencies) * 1.2)
            
            # 在柱状图上添加数值
            for bar, eff in zip(bars, efficiencies):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{eff:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # 保存性能图表
            perf_file = os.path.join(figures_dir, "performance_analysis.png")
            plt.savefig(perf_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  - 综合结果图: {plot_file}")
            print(f"  - 性能分析图: {perf_file}")
            
        except ImportError:
            print("  matplotlib未安装，跳过图表生成")
        except Exception as e:
            print(f"生成图表时出错: {e}")
    
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
        equilibrium_steps=1000,
        measurement_steps=5000,
        interaction_constant=1.0,
        random_seed=42,
        num_processes=4  # 使用4个进程
    )
    
    # 运行并行模拟
    results = simulator.run_simulation(
        temperature_range=(0.1, 2.5),
        num_temperatures=20
    )
    
    print("并行模拟完成！")
    print(f"结果已保存到: {simulator.base_output_dir}")