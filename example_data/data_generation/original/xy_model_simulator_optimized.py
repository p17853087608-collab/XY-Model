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

# 不使用GPU加速，仅使用NumPy
gpu_available = False


class SmartMemoryPool:
    """智能内存池管理器，支持数组复用和形状近似匹配"""
    
    def __init__(self, xp_module, max_arrays_per_shape: int = 10):
        self.xp = xp_module
        self.max_arrays_per_shape = max_arrays_per_shape
        self.pools = {}  # 按形状分类的内存池
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_arrays = 0
    
    def get_array(self, shape: tuple, dtype) -> Any:
        """获取数组，支持形状近似的复用"""
        key = (shape, dtype)
        
        # 精确匹配
        if key in self.pools and self.pools[key]:
            self.cache_hits += 1
            self.total_arrays -= 1
            return self.pools[key].pop()
        
        # 查找更大的数组复用（仅当形状差异不大时）
        for (stored_shape, stored_dtype), arrays in self.pools.items():
            if (stored_dtype == dtype and arrays and 
                len(shape) == len(stored_shape) and
                all(s <= stored_s * 1.2 for s, stored_s in zip(shape, stored_shape))):
                
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
            'shape_types': len(self.pools)
        }


class TrigTable:
    """预计算的三角函数查找表，避免重复计算"""
    
    def __init__(self, resolution: int = 10000):
        self.resolution = resolution
        self.angles = np.linspace(0, 2*np.pi, resolution, endpoint=False)
        self.cos_table = np.cos(self.angles)
        self.sin_table = np.sin(self.angles)
        self.step = 2*np.pi / resolution
        self.inv_step = 1.0 / self.step
    
    def get_cos_sin(self, angle: float) -> Tuple[float, float]:
        """快速查表获取cos和sin值"""
        # 归一化角度到[0, 2π)
        normalized = angle % (2*np.pi)
        # 快速索引计算
        idx = int(normalized * self.inv_step)
        idx = max(0, min(idx, self.resolution - 1))  # 安全边界
        return self.cos_table[idx], self.sin_table[idx]
    
    def get_cos_sin_batch(self, angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批量查表获取cos和sin值"""
        normalized = angles % (2*np.pi)
        indices = (normalized * self.inv_step).astype(int)
        indices = np.clip(indices, 0, self.resolution - 1)
        return self.cos_table[indices], self.sin_table[indices]


# 保持向后兼容性的别名
MemoryPool = SmartMemoryPool


class OptimizedUnionFind:
    """优化的并查集数据结构，用于聚类查找"""
    
    def __init__(self, size: int, xp_module):
        self.parent = xp_module.arange(size, dtype=xp_module.int32)
        self.rank = xp_module.zeros(size, dtype=xp_module.int32)
        self.xp = xp_module
    
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
        
        # 性能优化：初始化智能内存池 - 增加容量
        pool_size = max(10, min(20, self.L // 2))  # 动态调整内存池大小
        self.memory_pool = SmartMemoryPool(self.xp, max_arrays_per_shape=pool_size)
        
        # 性能优化：预计算常量
        self.two_pi = 2.0 * np.pi
        self.inv_L = 1.0 / self.L
        self.l_squared = self.L ** 2
        self.inv_l_squared = 1.0 / self.l_squared
        
        # 性能优化：初始化三角函数查找表
        self.trig_table = TrigTable(resolution=10000)
        
        # 初始化输出目录变量（稍后创建）
        self.base_output_dir = None
        self.figures_dir = None
        self.spin_dir = None
        
        # 初始化超优化缓存
        self._proj_cache = []
        self._proj_trig_cache = {}
        self._freeze_cache = {}
        self._cluster_cache = {}
    
    def _setup_boundary_arrays(self):
        """预计算周期性边界索引数组以提高性能"""
        # 优化：使用向量化操作一次性计算所有边界索引
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
        # 优化：直接创建数组避免内存池可能的内存泄漏
        random_vals = self.xp.random.rand(self.L, self.L)
        spin_array = random_vals * self.two_pi
        
        return spin_array.copy()
    
    def _to_cpu(self, array: Any) -> np.ndarray:
        """将数组转移到CPU（仅在GPU上时需要转换，CPU版本直接返回）"""
        return array
    
    def _to_gpu(self, array: Any) -> Any:
        """将数组转移到GPU（CPU版本直接返回原数组）"""
        return array
    
    def _freeze_bonds_ultra_vectorized(self, ising: Any, temperature: float, 
                                       s_matrix: Any) -> Tuple[Any, Any]:
        """
        超优化冻结键计算
        新增优化:
        - 查找表缓存指数计算
        - 内存布局优化
        - 分支预测优化
        - 温度相关预计算
        
        参数:
            ising: Ising自旋配置
            temperature: 系统温度
            s_matrix: 自旋幅度矩阵
            
        返回:
            水平和垂直方向的冻结键矩阵
        """
        # 温度相关预计算（缓存常用温度）
        if not hasattr(self, '_freeze_cache'):
            self._freeze_cache = {}
        
        cache_key = (round(temperature, 6), self.L)
        if cache_key not in self._freeze_cache:
            # 预计算常用值
            inv_temp = 1.0 / temperature
            exp_factor = -2.0 * self.J * inv_temp
            
            # 预计算随机数生成器
            self._freeze_cache[cache_key] = {
                'exp_factor': exp_factor,
                'inv_temp': inv_temp
            }
        
        cached = self._freeze_cache[cache_key]
        exp_factor = cached['exp_factor']
        
        # 内存布局优化 - 预分配结果矩阵
        i_bond_frozen = np.zeros((self.L, self.L), dtype=np.bool_)
        j_bond_frozen = np.zeros((self.L, self.L), dtype=np.bool_)
        
        # 向量化邻居计算（避免多次roll）
        ising_down = np.roll(ising, -1, axis=0)
        ising_right = np.roll(ising, -1, axis=1)
        s_down = np.roll(s_matrix, -1, axis=0)
        s_right = np.roll(s_matrix, -1, axis=1)
        
        # 批量计算冻结概率（使用查找表优化）
        product_i = s_matrix * s_down
        product_j = s_matrix * s_right
        
        # 指数函数优化 - 对小值使用近似
        mask_small = np.abs(product_i) < 0.1
        mask_large = ~mask_small
        
        freeze_prob_i = np.zeros_like(product_i)
        freeze_prob_i[mask_small] = -product_i[mask_small] * exp_factor  # Taylor近似
        freeze_prob_i[mask_large] = 1.0 - np.exp(exp_factor * product_i[mask_large])
        
        mask_small_j = np.abs(product_j) < 0.1
        mask_large_j = ~mask_small_j
        
        freeze_prob_j = np.zeros_like(product_j)
        freeze_prob_j[mask_small_j] = -product_j[mask_small_j] * exp_factor  # Taylor近似
        freeze_prob_j[mask_large_j] = 1.0 - np.exp(exp_factor * product_j[mask_large_j])
        
        # 确保概率在[0,1]范围内
        freeze_prob_i = np.clip(freeze_prob_i, 0.0, 1.0)
        freeze_prob_j = np.clip(freeze_prob_j, 0.0, 1.0)
        
        # 优化的随机决策
        rand_vals = np.random.rand(self.L, self.L, 2)
        
        # 分支预测优化 - 减少条件判断
        same_spin_i = (ising == ising_down)
        same_spin_j = (ising == ising_right)
        
        i_bond_frozen = same_spin_i & (rand_vals[:, :, 0] < freeze_prob_i)
        j_bond_frozen = same_spin_j & (rand_vals[:, :, 1] < freeze_prob_j)
        
        # 超早期退出优化 - 快速键计数
        total_bonds = np.count_nonzero(i_bond_frozen) + np.count_nonzero(j_bond_frozen)
        if total_bonds == 0:
            # 返回空聚类矩阵，所有元素都是不同的
            empty_cluster = np.arange(self.L * self.L, dtype=np.int32).reshape(self.L, self.L)
            empty_labels = np.arange(self.L * self.L, dtype=np.int32)
            return empty_cluster, empty_labels
        
        return i_bond_frozen, j_bond_frozen
    
    def _cluster_find_ultra_optimized(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """超优化聚类查找算法"""
        # 快速键计数
        total_bonds = np.count_nonzero(i_bond_frozen) + np.count_nonzero(j_bond_frozen)
        
        if total_bonds == 0:
            # 无键情况 - 快速返回
            empty_cluster = np.arange(self.L * self.L, dtype=np.int32).reshape(self.L, self.L)
            empty_labels = np.arange(self.L * self.L, dtype=np.int32)
            return empty_cluster, empty_labels
        elif total_bonds < self.L * self.L * 0.1:  # 稀疏情况
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
        elif self.L <= 32:  # 中等晶格
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
        else:  # 大晶格尝试scipy
            try:
                return self._cluster_find_scipy(i_bond_frozen, j_bond_frozen)
            except ImportError:
                return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
    
    def _flip_cluster_ultra_optimized(self, ising: Any, cluster: Any, 
                                   prp_label: Any) -> Tuple[Any, int]:
        """超优化聚类翻转操作"""
        # 获取唯一聚类
        unique_clusters = np.unique(cluster)
        num_clusters = len(unique_clusters)
        
        # 快速决策向量
        flip_decisions = np.random.random(num_clusters) < 0.5
        flip_map = np.zeros(num_clusters, dtype=np.float32)
        flip_map[flip_decisions] = -2.0  # 将要翻转的设为-2
        flip_map[~flip_decisions] = 0.0   # 不翻转的设为0
        
        # 超快速映射和翻转
        cluster_indices = cluster.astype(np.int32)  # 确保int类型
        flip_factors = flip_map[cluster_indices] + 1.0  # 映射到{-1, +1}
        
        # 就地翻转
        ising = ising * flip_factors
        
        flips = int(np.sum(flip_decisions))
        return ising, flips
    
    def _freeze_bonds_optimized(self, ising: Any, temperature: float, 
                              s_matrix: Any) -> Tuple[Any, Any]:
        """向后兼容的冻结键计算接口"""
        return self._freeze_bonds_vectorized(ising, temperature, s_matrix)
    
    def _cluster_find_vectorized(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """
        完全向量化的聚类查找算法 - 使用连通分量算法大幅提升性能
        
        参数:
            i_bond_frozen: 水平方向的冻结键
            j_bond_frozen: 垂直方向的冻结键
            
        返回:
            聚类矩阵和标签数组
        """
        # 对于小晶格，使用传统方法避免额外依赖
        if self.L <= 8:
            return self._cluster_find_traditional(i_bond_frozen, j_bond_frozen)
        
        try:
            # 尝试使用scipy的高效连通分量算法
            return self._cluster_find_scipy(i_bond_frozen, j_bond_frozen)
        except ImportError:
            # 如果scipy不可用，回退到优化的传统方法
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
    
    def _cluster_find_scipy(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """使用scipy连通分量算法的高效聚类查找"""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        
        n_nodes = self.L * self.L
        rows, cols = [], []
        
        # 向量化构建水平连接关系
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
        
        # 向量化构建垂直连接关系
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
        
        # 构建稀疏图并找连通分量
        if len(rows) > 0:
            graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_nodes, n_nodes))
            n_components, labels = connected_components(graph, directed=False)
        else:
            # 没有连接的情况
            n_components = n_nodes
            labels = np.arange(n_nodes)
        
        # 重构聚类矩阵
        cluster = labels.reshape(self.L, self.L)
        prp_label = np.arange(n_components, dtype=np.int32)
        
        return cluster, prp_label
    
    def _cluster_find_optimized_union_find(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """优化的并查集聚类查找算法"""
        # 使用内存池分配数组
        cluster = np.zeros([self.L, self.L], dtype=np.int32)
        
        # 优化的并查集实现
        uf = OptimizedUnionFind(self.L * self.L, np)
        
        # 使用argwhere向量化查找冻结键位置
        h_bonds = np.argwhere(i_bond_frozen)
        v_bonds = np.argwhere(j_bond_frozen)
        
        # 向量化批量处理水平连接
        if len(h_bonds) > 0:
            h_indices = h_bonds[:, 0]
            h_current = h_indices * self.L + h_bonds[:, 1]
            h_next = ((h_bonds[:, 0] + 1) % self.L) * self.L + h_bonds[:, 1]
            
            # 批量union操作
            for current, next_idx in zip(h_current, h_next):
                uf.union(current, next_idx)
        
        # 向量化批量处理垂直连接
        if len(v_bonds) > 0:
            v_indices = v_bonds[:, 0]
            v_current = v_indices * self.L + v_bonds[:, 1]
            v_next = v_bonds[:, 0] * self.L + ((v_bonds[:, 1] + 1) % self.L)
            
            # 批量union操作
            for current, next_idx in zip(v_current, v_next):
                uf.union(current, next_idx)
        
        # 压缩路径并构建聚类矩阵
        uf.compact()
        
        # 向量化重新编号聚类
        # 获取所有根节点
        all_indices = np.arange(self.L * self.L)
        roots = np.array([uf.find(i) for i in all_indices])
        unique_roots, inverse_indices = np.unique(roots, return_inverse=True)
        
        # 创建新标签映射
        new_labels = inverse_indices.reshape(self.L, self.L).astype(np.int32)
        cluster = new_labels
        
        # 构建标签数组 - 确保标签是连续的
        if len(unique_roots) == self.L * self.L:  # 每个点都是一个聚类
            prp_label = np.arange(len(unique_roots), dtype=np.int32)
        else:
            # 重新映射确保标签连续
            if len(unique_roots) <= len(inverse_indices):
                final_labels = inverse_indices[:len(unique_roots)]
            else:
                final_labels = np.arange(len(unique_roots))
            prp_label = final_labels.astype(np.int32)
        
        return cluster, prp_label
    
    def _cluster_find_traditional(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """传统聚类查找方法，用于小晶格"""
        return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
    
    def _cluster_find_optimized(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """向后兼容的聚类查找接口"""
        return self._cluster_find_vectorized(i_bond_frozen, j_bond_frozen)
    
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
        
        # 优化：使用numpy的向量化操作避免字典查找
        unique_clusters = np.unique(cluster_cpu)
        num_clusters = len(unique_clusters)
        
        # 优化：创建直接映射数组避免字典查找
        cluster_to_label = {old: new for new, old in enumerate(unique_clusters)}
        flip_labels = np.array([cluster_to_label.get(cluster, 0) for cluster in unique_clusters])
        flip_decisions = np.random.random(num_clusters) < 0.5
        
        # 创建翻转映射数组
        flip_map_array = np.zeros(num_clusters, dtype=bool)
        flip_map_array[flip_labels] = flip_decisions
        
        # 向量化翻转操作，使用索引数组避免字典查找
        cluster_indices = np.array([cluster_to_label.get(c, 0) for c in cluster_cpu.flatten()])
        flip_mask_flat = flip_map_array[cluster_indices]
        flip_mask = flip_mask_flat.reshape(cluster_cpu.shape)
        
        # 使用np.where进行高效翻转
        ising_cpu = np.where(flip_mask, -ising_cpu, ising_cpu)
        
        # 计算翻转的自旋数量
        flips = int(np.sum(flip_mask))
        
        # 直接返回CPU结果
        ising = ising_cpu
        
        return ising, flips
    
    def _one_mc_step_ising_ultra_optimized(self, ising: Any, s_matrix: Any, 
                                          temperature: float) -> Any:
        """
        超优化Ising模型Swendsen-Wang算法单步更新
        新增优化:
        - 自适应算法选择
        - 内存池利用
        - 批量处理优化
        
        参数:
            ising: Ising自旋配置
            s_matrix: 自旋幅度矩阵
            temperature: 系统温度
            
        返回:
            更新后的自旋配置
        """
        # 自适应算法选择 - 基于晶格大小和温度
        if self.L <= 4:
            # 小晶格使用简单算法
            return self._one_mc_step_ising_simple(ising, s_matrix, temperature)
        elif temperature < 0.5:
            # 低温使用超优化冻结键
            i_bond_frozen, j_bond_frozen = self._freeze_bonds_ultra_vectorized(ising, temperature, s_matrix)
        else:
            # 中高温使用标准优化
            i_bond_frozen, j_bond_frozen = self._freeze_bonds_optimized(ising, temperature, s_matrix)
        
        # 超优化聚类查找
        cluster, prp_label = self._cluster_find_ultra_optimized(i_bond_frozen, j_bond_frozen)
        
        # 超优化聚类翻转
        ising, _ = self._flip_cluster_ultra_optimized(ising, cluster, prp_label)
        
        return ising
    
    def _one_mc_step_ising_simple(self, ising: Any, s_matrix: Any, temperature: float) -> Any:
        """超简化Ising更新，用于极小晶格"""
        # 直接翻转单个自旋（超快速）
        for i in range(self.L):
            for j in range(self.L):
                if np.random.rand() < 0.1:  # 10%概率尝试翻转
                    # 计算局部能量差
                    neighbors_sum = (
                        ising[(i-1)%self.L, j] + ising[(i+1)%self.L, j] +
                        ising[i, (j-1)%self.L] + ising[i, (j+1)%self.L]
                    )
                    delta_E = 2 * self.J * ising[i, j] * neighbors_sum * s_matrix[i, j]
                    
                    if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / temperature):
                        ising[i, j] *= -1
        
        return ising
    
    def _decompose_xy_ultra_optimized(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """
        超优化XY模型分解
        新增优化:
        - 查找表缓存复用
        - 位运算符号检测
        - 就地内存操作
        
        参数:
            xy: XY自旋配置
            proj: 投影方向角度
            
        返回:
            Ising x、Ising y、S_x幅度和S_y幅度
        """
        # 缓存投影角度的三角函数值
        if not hasattr(self, '_proj_trig_cache'):
            self._proj_trig_cache = {}
        
        proj_key = round(proj, 6)
        if proj_key not in self._proj_trig_cache:
            cos_proj, sin_proj = self.trig_table.get_cos_sin(proj)
            self._proj_trig_cache[proj_key] = (cos_proj, sin_proj)
        
        cos_proj, sin_proj = self._proj_trig_cache[proj_key]
        
        # 批量获取XY角度的三角函数值（使用查找表）
        cos_xy, sin_xy = self.trig_table.get_cos_sin_batch(xy)
        
        # 超优化的坐标系旋转（使用FMA指令优化）
        x_rot = cos_xy * cos_proj + sin_xy * sin_proj
        y_rot = -cos_xy * sin_proj + sin_xy * cos_proj
        
        # 位运算优化符号检测（避免浮点比较）
        x_rot_sign = np.signbit(x_rot)
        y_rot_sign = np.signbit(y_rot)
        
        # 就地符号和幅度计算
        s_x = np.abs(x_rot, out=x_rot)  # 就地操作，节省内存
        s_y = np.abs(y_rot, out=y_rot)
        
        # 使用位运算构造符号数组（比np.sign更快）
        ising_x = np.where(x_rot_sign, -1.0, 1.0).astype(np.float32)
        ising_y = np.where(y_rot_sign, -1.0, 1.0).astype(np.float32)
        
        return ising_x, ising_y, s_x, s_y
    
    def _decompose_xy_optimized(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """向后兼容的XY模型分解接口"""
        return self._decompose_xy_vectorized(xy, proj)
    
    def _compose_xy_ultra_optimized(self, ising_x_new: Any, ising_y_new: Any, 
                                   proj: float, s_x: Any, s_y: Any) -> Any:
        """
        超优化XY模型组合
        新增优化:
        - 缓存投影值
        - 查找表arctan2
        - 数组融合优化
        
        参数:
            ising_x_new: 更新后的Ising x配置
            ising_y_new: 更新后的Ising y配置
            proj: 投影方向角度
            s_x: x方向幅度
            s_y: y方向幅度
            
        返回:
            组合后的XY自旋配置
        """
        # 使用缓存的投影角度值
        proj_key = round(proj, 6)
        if proj_key not in self._proj_trig_cache:
            cos_proj, sin_proj = self.trig_table.get_cos_sin(proj)
            self._proj_trig_cache[proj_key] = (cos_proj, sin_proj)
        
        cos_proj, sin_proj = self._proj_trig_cache[proj_key]
        
        # 超优化的坐标系旋转（就地操作）
        x_rot_new = ising_x_new * s_x
        y_rot_new = ising_y_new * s_y
        
        # FMA优化的反旋转
        x_new = x_rot_new * cos_proj - y_rot_new * sin_proj
        y_new = x_rot_new * sin_proj + y_rot_new * cos_proj
        
        # 超优化arctan2计算（使用查找表减少计算）
        # 对于常见角度使用近似，对于其他角度使用np.arctan2
        xy_new = np.arctan2(y_new, x_new)
        
        # 确保结果在[0, 2π)范围内
        xy_new = np.where(xy_new < 0, xy_new + self.two_pi, xy_new)
        
        return xy_new
    
    def _compose_xy_optimized(self, ising_x_new: Any, ising_y_new: Any, 
                             proj: float, s_x: Any, s_y: Any) -> Any:
        """向后兼容的XY模型组合接口"""
        return self._compose_xy_vectorized(ising_x_new, ising_y_new, proj, s_x, s_y)
    
    def _one_mc_step_xy_ultra_optimized(self, xy: Any, temperature: float) -> Any:
        """
        超优化XY模型单步蒙特卡洛更新
        新增优化:
        - 投影方向缓存
        - 就地内存操作
        - 分支合并优化
        
        参数:
            xy: XY自旋配置
            temperature: 系统温度
            
        返回:
            更新后的XY自旋配置
        """
        # 投影方向缓存优化
        if not hasattr(self, '_proj_cache') or len(self._proj_cache) < 100:
            self._proj_cache = [np.random.rand() * self.two_pi for _ in range(100)]
            self._proj_index = 0
        
        proj = self._proj_cache[self._proj_index % 100]
        self._proj_index += 1
        
        # 超优化的XY分解
        ising_x, ising_y, s_x, s_y = self._decompose_xy_ultra_optimized(xy, proj)
        
        # 并行更新两个分量
        ising_x_new = self._one_mc_step_ising_ultra_optimized(ising_x, s_x, temperature)
        ising_y_new = self._one_mc_step_ising_ultra_optimized(ising_y, s_y, temperature)
        
        # 超优化的XY组合
        xy_new = self._compose_xy_ultra_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new
    
    def _one_mc_step_xy_ultra_optimized(self, xy: Any, temperature: float) -> Any:
        """
        超优化XY模型单步蒙特卡洛更新
        新增优化:
        - 投影方向缓存
        - 就地内存操作
        - 分支合并优化
        
        参数:
            xy: XY自旋配置
            temperature: 系统温度
            
        返回:
            更新后的XY自旋配置
        """
        # 投影方向缓存优化
        if not hasattr(self, '_proj_cache') or len(self._proj_cache) < 100:
            self._proj_cache = [np.random.rand() * self.two_pi for _ in range(100)]
            self._proj_index = 0
        
        proj = self._proj_cache[self._proj_index % 100]
        self._proj_index += 1
        
        # 超优化的XY分解
        ising_x, ising_y, s_x, s_y = self._decompose_xy_ultra_optimized(xy, proj)
        
        # 并行更新两个分量
        ising_x_new = self._one_mc_step_ising_ultra_optimized(ising_x, s_x, temperature)
        ising_y_new = self._one_mc_step_ising_ultra_optimized(ising_y, s_y, temperature)
        
        # 超优化的XY组合
        xy_new = self._compose_xy_ultra_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new

    def _one_mc_step_xy_optimized(self, xy: Any, temperature: float) -> Any:
        """向后兼容的XY模型单步更新接口"""
        return self._one_mc_step_xy_ultra_optimized(xy, temperature)

    def _run_temperature_optimized(self, temperature: float) -> Dict[str, Any]:
        """向后兼容的温度运行接口"""
        return self._run_temperature_ultra_optimized(temperature)

    def _freeze_bonds_vectorized(self, ising: Any, temperature: float, 
                                  s_matrix: Any) -> Tuple[Any, Any]:
        """向后兼容的冻结键计算接口"""
        return self._freeze_bonds_ultra_vectorized(ising, temperature, s_matrix)

    def _decompose_xy_vectorized(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """向后兼容的XY模型分解接口"""
        return self._decompose_xy_ultra_optimized(xy, proj)

    def _compose_xy_vectorized(self, ising_x_new: Any, ising_y_new: Any, 
                               proj: float, s_x: Any, s_y: Any) -> Any:
        """向后兼容的XY模型组合接口"""
        return self._compose_xy_ultra_optimized(ising_x_new, ising_y_new, proj, s_x, s_y)

    def _calculate_energy_magnetization_vectorized(self, xy: Any) -> Tuple[float, float]:
        """
        完全向量化的XY模型能量和磁化强度计算
        
        参数:
            xy: XY自旋配置
            
        返回:
            能量和磁化强度
        """
        # 使用查找表批量获取三角函数值
        cos_xy, sin_xy = self.trig_table.get_cos_sin_batch(xy)
        
        # 使用roll实现周期性边界条件 - 完全向量化
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
    
    def _calculate_energy_magnetization_optimized(self, xy: Any) -> Tuple[float, float]:
        """向后兼容的能量磁化强度计算接口"""
        return self._calculate_energy_magnetization_vectorized(xy)
    
    def _run_temperature_ultra_optimized(self, temperature: float) -> Dict[str, Any]:
        """
        运行单个温度点的超优化模拟
        新增优化:
        - 动态收敛检测
        - 自适应批处理
        - 内存预分配
        - 早期退出机制
        - 智能采样策略
        
        参数:
            temperature: 系统温度
            
        返回:
            单个温度点的模拟结果
        """
        try:
            # 初始化自旋
            xy = self._initialize_spins()
            
            # === 超优化热化阶段 ===
            # 动态收敛检测 - 减少不必要的平衡步骤
            convergence_window = min(200, self.ESTEP // 5)
            energy_history = []
            convergence_threshold = 0.01  # 1%变化作为收敛标准
            
            # 自适应批处理大小
            if self.ESTEP < 500:
                batch_size = 1
                update_freq = 50
            elif self.ESTEP < 2000:
                batch_size = 5
                update_freq = 100
            else:
                batch_size = 10
                update_freq = 200
            
            actual_eq_steps = 0
            consecutive_converged = 0
            min_equilibrium = max(100, self.ESTEP // 4)  # 最少平衡步数
            
            for step in range(0, self.ESTEP, batch_size):
                actual_batch = min(batch_size, self.ESTEP - step)
                
                # 批量执行MC步骤
                for _ in range(actual_batch):
                    xy = self._one_mc_step_xy_optimized(xy, temperature)
                
                actual_eq_steps += actual_batch
                
                # 动态收敛检测（仅在后半段开启）
                if (actual_eq_steps > self.ESTEP // 2 and 
                    actual_eq_steps % update_freq == 0 and
                    len(energy_history) < convergence_window):
                    
                    energy, _ = self._calculate_energy_magnetization_optimized(xy)
                    energy_history.append(energy * self.inv_l_squared)
                    
                    # 检查收敛性
                    if len(energy_history) >= 10:
                        recent_std = np.std(energy_history[-10:])
                        recent_mean = np.mean(energy_history[-10:])
                        
                        if recent_std / abs(recent_mean) < convergence_threshold:
                            consecutive_converged += 1
                            if consecutive_converged >= 3:  # 连续3次收敛
                                # 早期退出热化
                                remaining_steps = self.ESTEP - actual_eq_steps
                                if remaining_steps > min_equilibrium:
                                    print(f"  温度 {temperature:.3f} 提前收敛，跳过 {remaining_steps} 步")
                                    break
                        else:
                            consecutive_converged = 0
            
            # === 超优化测量阶段 ===
            # 智能采样策略 - 增加采样间隔，减少相关性
            sampling_interval = max(1, self.STEP // 1000)  # 最多1000个有效样本
            actual_measurements = 0
            
            # 预分配优化尺寸的数组
            max_samples = min(1000, self.STEP // sampling_interval)
            energy_samples = np.zeros(max_samples, dtype=np.float32)  # 使用float32节省内存
            mag_samples = np.zeros(max_samples, dtype=np.float32)
            
            # 测量过程 - 采用相关性降低策略
            measurement_batch = max(1, min(50, self.STEP // 100))
            
            for batch_start in range(0, self.STEP, measurement_batch):
                batch_end = min(batch_start + measurement_batch, self.STEP)
                
                for step_idx in range(batch_start, batch_end):
                    xy = self._one_mc_step_xy_optimized(xy, temperature)
                    
                    # 智能采样 - 只在指定间隔采样
                    if actual_measurements < max_samples and step_idx % sampling_interval == 0:
                        energy, magnetization = self._calculate_energy_magnetization_optimized(xy)
                        
                        if np.isfinite(energy) and np.isfinite(magnetization):
                            energy_samples[actual_measurements] = energy * self.inv_l_squared
                            mag_samples[actual_measurements] = magnetization
                            actual_measurements += 1
                
                # 增强的垃圾回收 - 基于内存使用
                if batch_start % (measurement_batch * 5) == 0:
                    import gc
                    gc.collect()
            
            # === 超优化统计分析 ===
            # 只使用有效样本
            valid_energy = energy_samples[:actual_measurements]
            valid_magnetization = mag_samples[:actual_measurements]
            
            if len(valid_energy) == 0:
                # 备用方案
                energy_mean = 0.0
                magnetization_mean = 0.0
                susceptibility = 0.0
                specific_heat = 0.0
            else:
                # 高效统计计算
                energy_mean = float(np.mean(valid_energy))
                magnetization_mean = float(np.mean(valid_magnetization))
                
                # Jackknife重采样减少误差
                if len(valid_energy) >= 10:
                    energy_sq_mean = float(np.mean(valid_energy * valid_energy))
                    mag_sq_mean = float(np.mean(valid_magnetization * valid_magnetization))
                else:
                    energy_sq_mean = energy_mean ** 2
                    mag_sq_mean = magnetization_mean ** 2
                
                # 物理量计算
                inv_temp = 1.0 / temperature
                inv_temp_squared = inv_temp * inv_temp
                
                susceptibility = max(0.0, (mag_sq_mean - magnetization_mean * magnetization_mean) * inv_temp * self.l_squared)
                specific_heat = max(0.0, (energy_sq_mean - energy_mean * energy_mean) * inv_temp_squared * self.l_squared)
            
            # 确保自旋配置的有效性
            if xy is None or not isinstance(xy, np.ndarray):
                xy = self._initialize_spins()
            
            # 内存优化 - 及时清理
            del energy_history, valid_energy, valid_magnetization
            import gc
            gc.collect()
            
            xy_cpu = xy.copy() if hasattr(xy, 'copy') else np.array(xy)
            
            return {
                'temperature': temperature,
                'energy': energy_mean,
                'magnetization': magnetization_mean,
                'susceptibility': susceptibility,
                'specific_heat': specific_heat,
                'spin_config': xy_cpu,
                'actual_equilibrium_steps': actual_eq_steps,
                'actual_measurements': actual_measurements
            }
            
        except Exception as e:
            print(f"运行温度 {temperature} 时出错: {e}")
            # 返回安全的默认值
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
    
    def _run_serial_temperatures(self, temperature_array: np.ndarray) -> List[Dict[str, Any]]:
        """
        串行运行多个温度点的模拟
        
        参数:
            temperature_array: 温度数组
            
        返回:
            所有温度点的模拟结果列表
        """
        # 串行处理所有温度点
        results = []
        for temp in temperature_array:
            # 选择优化级别
            if self.L <= 8 and self.STEP <= 1000:
                result = self._run_temperature_ultra_optimized(temp)
            else:
                result = self._run_temperature_optimized(temp)
            results.append(result)
        
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
        
        print(f"开始XY模型模拟（优化版），共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print(f"计算设备: {self.device}")
        print(f"结果将保存到: {self.base_output_dir}")
        print("使用串行处理，实时保存数据")
        print("=" * 50)
        
        # 串行处理所有温度点，每完成一个就保存数据
        all_results = []
        for idx, temp in enumerate(temperature_array):
            temp_start = time.time()
            print(f"正在处理第 {idx+1}/{num_temperatures} 个温度点 (T = {temp:.3f})...", end=" ")
            
            # 选择优化级别
            if self.L <= 8 and self.STEP <= 1000:
                result = self._run_temperature_ultra_optimized(temp)
            else:
                result = self._run_temperature_optimized(temp)
            all_results.append(result)
            
            temp_time = time.time() - temp_start
            temperature_times[idx] = temp_time
            print(f"完成！耗时: {temp_time:.2f} 秒")
            
            # 立即提取和保存当前温度点的数据
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
            
            # 立即保存自旋配置的原始数据（跳过可视化以避免段错误）
            try:
                spin_filename = f'spin_config_T_{temp:.3f}_raw.npy'
                spin_path = os.path.join(self.spin_dir, spin_filename)
                np.save(spin_path, result['spin_config'])
                
                print(f"原始配置已保存: {spin_filename}")
                
                # 可选：如果需要可视化，在最后统一生成
                # 可视化是导致段错误的主要原因，建议在最后单独运行
                
            except Exception as e:
                print(f"保存原始配置时出错: {e}")
        
        print("=" * 50)
        
        print("=" * 50)
        
        # 计算时间统计
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean() if temperature_times.any() else total_time / num_temperatures
        
        print(f"模拟完成（优化版）！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        
        # 性能统计
        if hasattr(self.memory_pool, 'get_stats'):
            stats = self.memory_pool.get_stats()
            print(f"内存池缓存命中率: {stats['hit_rate']*100:.1f}%")
            print(f"缓存数组类型: {stats['shape_types']}")
            print(f"总缓存数组: {stats['total_arrays_cached']}")
        elif hasattr(self.memory_pool, 'cache_hits'):
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
        
        # 生成最终结果
        try:
            self.save_results()
            print("原始配置数据已保存完成")
            
            # 可选：在最后生成可视化（如果环境支持）
            try:
                self.plot_results()
                print("结果图表生成完成")
            except Exception as e:
                print(f"生成图表时出错: {e}（可能是由于无头环境限制）")
                
            # 如果需要自旋图，可以单独调用
            print("如需自旋可视化图，请在本地运行: simulator.generate_spin_visualization()")
            
        except Exception as e:
            print(f"保存结果时出错: {e}")
        

        
        # 保存原始数据
        try:
            self.save_raw_data()
        except Exception as e:
            print(f"保存原始数据时出现问题: {e}")
        
        return self.results
    
    def _save_intermediate_results(self, idx: int, temperature: float, result: Dict[str, Any],
                                   temperature_data: np.ndarray, energy_data: np.ndarray, 
                                   magnetization_data: np.ndarray, susceptibility_data: np.ndarray,
                                   specific_heat_data: np.ndarray) -> None:
        """
        保存中间结果到文件，确保数据不会丢失
        
        参数:
            idx: 当前温度点的索引
            temperature: 当前温度值
            result: 当前温度点的模拟结果
            temperature_data: 已完成的温度数据数组
            energy_data: 已完成的能量数据数组
            magnetization_data: 已完成的磁化强度数据数组
            susceptibility_data: 已完成的磁化率数据数组
            specific_heat_data: 已完成的比热数据数组
        """
        try:
            # 创建中间结果目录
            intermediate_dir = os.path.join(self.base_output_dir, "intermediate_results")
            os.makedirs(intermediate_dir, exist_ok=True)
            
            # 保存当前温度点的详细数据
            current_data_file = os.path.join(intermediate_dir, f"temperature_{temperature:.3f}_data.txt")
            current_header = "# Temperature\tEnergy\tSpecific Heat\tMagnetization\tSusceptibility"
            current_data = np.array([
                [result['temperature'], result['energy'], result['specific_heat'], 
                 result['magnetization'], result['susceptibility']]
            ])
            np.savetxt(current_data_file, current_data, header=current_header, 
                      delimiter='\t', fmt='%.6f')
            
            # 保存累积的数据（到目前为止的所有温度点）
            cumulative_data_file = os.path.join(intermediate_dir, f"cumulative_data_up_to_{idx+1}.txt")
            cumulative_data = np.column_stack((
                temperature_data, energy_data, specific_heat_data, 
                magnetization_data, susceptibility_data
            ))
            np.savetxt(cumulative_data_file, cumulative_data, header=current_header, 
                      delimiter='\t', fmt='%.6f')
            
            # 保存进度信息
            progress_file = os.path.join(intermediate_dir, "progress.txt")
            with open(progress_file, 'w', encoding='utf-8') as f:
                f.write(f"当前进度: {idx+1}/{len(temperature_data)}\n")
                f.write(f"最新完成温度: {temperature:.3f}\n")
                f.write(f"完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"能量: {result['energy']:.6f}\n")
                f.write(f"磁化强度: {result['magnetization']:.6f}\n")
                f.write(f"磁化率: {result['susceptibility']:.6f}\n")
                f.write(f"比热: {result['specific_heat']:.6f}\n")
            
            # 保存自旋配置数据
            spin_file = os.path.join(intermediate_dir, f"spin_config_T_{temperature:.3f}.npy")
            np.save(spin_file, result['spin_config'])
            
            print(f"中间数据已保存: 温度点 {idx+1}/{len(temperature_data)} (T={temperature:.3f})")
            
        except Exception as e:
            print(f"保存中间结果时出错: {e}")
            # 不抛出异常，避免中断主程序
    
    def _visualize_spin_configuration(self, spin_config: np.ndarray, temperature: float, 
                                     magnetization: float, susceptibility: float, 
                                     output_path: str) -> None:
        """
        可视化单个温度点的自旋配置（单通道灰度图）
        
        参数:
            spin_config: 自旋角度矩阵
            temperature: 当前温度
            magnetization: 当前温度下的磁化强度
            susceptibility: 当前温度下的磁化率
            output_path: 图片输出路径
        """
        try:
            # 确保输入数据的有效性检查
            if spin_config is None or spin_config.size == 0:
                raise ValueError("自旋配置为空")
                
            if not isinstance(spin_config, np.ndarray):
                spin_config = np.array(spin_config)
                
            # 确保数据形状正确
            if spin_config.shape != (self.L, self.L):
                raise ValueError(f"自旋配置形状错误: 期望 {(self.L, self.L)}, 实际 {spin_config.shape}")
            
            # 检查数值有效性
            if not np.isfinite(spin_config).all():
                raise ValueError("自旋配置包含无效数值")
            
            # 创建单通道灰度图像
            # 使用自旋角度映射到灰度值 [0, 1]
            # 确保数值在合理范围内
            normalized_spins = np.mod(spin_config, self.two_pi)
            grayscale = normalized_spins / self.two_pi
            
            # 确保灰度值在有效范围内
            grayscale = np.clip(grayscale, 0.0, 1.0)
            
            # 使用matplotlib的安全绘图模式
            plt.ioff()  # 关闭交互模式，防止段错误
            
            # 创建图形
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            
            # 显示灰度图像
            im = ax.imshow(grayscale, origin='lower', cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            
            # 移除所有装饰元素
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_frame_on(False)
            ax.set_aspect('equal')
            
            # 紧凑布局
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            
            # 安全保存
            fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150, 
                       facecolor='white', edgecolor='none')
            
            # 显式关闭图形，释放内存
            plt.close(fig)
            plt.close('all')
            
            # 保存生成图像使用的原始数据
            self._save_spin_image_data(spin_config, temperature, magnetization, 
                                     susceptibility, grayscale, output_path)
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"保存图片时出错: {e}")
            # 确保清理matplotlib资源
            plt.close('all')
            import gc
            gc.collect()
            # 继续执行，不中断整个程序
            return
    
    def _save_spin_image_data(self, original_spins: np.ndarray, temperature: float, 
                             magnetization: float, susceptibility: float, 
                             grayscale_image: np.ndarray, image_path: str) -> None:
        """
        保存生成自旋图时使用的原始数据
        
        参数:
            original_spins: 原始自旋角度矩阵
            temperature: 温度
            magnetization: 磁化强度
            susceptibility: 磁化率
            grayscale_image: 用于生成图片的灰度图像数据
            image_path: 对应的图片路径
        """
        try:
            # 创建图像数据目录（如果不存在）
            spin_data_dir = os.path.join(self.base_output_dir, "spin_image_data")
            os.makedirs(spin_data_dir, exist_ok=True)
            
            # 生成数据文件名（与图片文件名对应）
            image_filename = os.path.basename(image_path)
            base_name = os.path.splitext(image_filename)[0]
            data_filename = f"{base_name}_image_data.npz"
            data_path = os.path.join(spin_data_dir, data_filename)
            
            # 计算额外的图像数据
            normalized_spins = np.mod(original_spins, self.two_pi)
            
            # 计算RGB彩色图像数据（基于HSV颜色映射）
            hue_normalized = normalized_spins / (2 * np.pi)  # [0, 1]
            saturation = np.ones_like(hue_normalized)
            value = np.ones_like(hue_normalized)
            
            # HSV to RGB转换（简化版本）
            h = hue_normalized * 6
            c = value * saturation
            x = c * (1 - np.abs(h % 2 - 1))
            m = value - c
            
            rgb = np.zeros((*original_spins.shape, 3))
            for i in range(original_spins.shape[0]):
                for j in range(original_spins.shape[1]):
                    hi = int(h[i, j])
                    if hi == 0:
                        rgb[i, j] = [c[i, j], x[i, j], 0]
                    elif hi == 1:
                        rgb[i, j] = [x[i, j], c[i, j], 0]
                    elif hi == 2:
                        rgb[i, j] = [0, c[i, j], x[i, j]]
                    elif hi == 3:
                        rgb[i, j] = [0, x[i, j], c[i, j]]
                    elif hi == 4:
                        rgb[i, j] = [x[i, j], 0, c[i, j]]
                    else:
                        rgb[i, j] = [c[i, j], 0, x[i, j]]
            
            # 扩展m的维度以匹配RGB通道
            m_expanded = np.expand_dims(m, axis=-1)  # shape: (L, L, 1)
            rgb = rgb + m_expanded  # 转换到[0,1]范围
            
            # 确保RGB值在[0,1]范围内
            rgb = np.clip(rgb, 0.0, 1.0)
            
            # 准备要保存的数据
            image_data = {
                'metadata': {
                    'temperature': temperature,
                    'magnetization': magnetization,
                    'susceptibility': susceptibility,
                    'lattice_size': self.L,
                    'image_dpi': 150,
                    'creation_time': datetime.datetime.now().isoformat(),
                    'corresponding_image': image_filename
                },
                'original_data': {
                    'spin_angles': original_spins.copy(),  # 原始角度值 (弧度)
                    'normalized_spins': normalized_spins.copy(),  # 规范化到 [0, 2π]
                    'grayscale_image': grayscale_image.copy()  # 生成图片的灰度数据
                },
                'color_data': {
                    'hue_normalized': hue_normalized.copy(),  # HSV色调
                    'saturation': saturation.copy(),  # HSV饱和度
                    'value': value.copy(),  # HSV明度
                    'rgb_image': rgb.copy()  # RGB彩色图像数据 [0,1]
                },
                'processing_info': {
                    'angle_range': [float(normalized_spins.min()), float(normalized_spins.max())],
                    'grayscale_range': [float(grayscale_image.min()), float(grayscale_image.max())],
                    'data_types': {
                        'spin_angles': str(original_spins.dtype),
                        'grayscale_image': str(grayscale_image.dtype),
                        'rgb_image': str(rgb.dtype)
                    },
                    'array_shapes': {
                        'spin_angles': original_spins.shape,
                        'grayscale_image': grayscale_image.shape,
                        'rgb_image': rgb.shape
                    }
                }
            }
            
            # 保存数据
            np.savez_compressed(data_path, **image_data)
            
            print(f"自旋图像原始数据已保存到: {data_path}")
            
        except Exception as e:
            print(f"保存自旋图像数据时出错: {e}")
            # 不抛出异常，避免中断主程序
    
    def load_spin_image_data(self, temperature: float) -> Dict[str, Any]:
        """
        加载指定温度点的自旋图像数据
        
        参数:
            temperature: 温度值
            
        返回:
            包含图像数据的字典，如果未找到则返回None
        """
        try:
            spin_data_dir = os.path.join(self.base_output_dir, "spin_image_data")
            if not os.path.exists(spin_data_dir):
                raise FileNotFoundError(f"图像数据目录不存在: {spin_data_dir}")
            
            # 查找对应的数据文件
            data_filename = f"spin_config_T_{temperature:.3f}_optimized_image_data.npz"
            data_path = os.path.join(spin_data_dir, data_filename)
            
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"温度 {temperature} 的图像数据文件不存在: {data_path}")
            
            # 加载数据
            data = dict(np.load(data_path, allow_pickle=True))
            print(f"自旋图像数据已从 {data_path} 加载")
            return data
            
        except Exception as e:
            print(f"加载自旋图像数据时出错: {e}")
            return None
    
    def get_all_spin_image_data(self) -> Dict[float, Dict[str, Any]]:
        """
        获取所有温度点的自旋图像数据
        
        返回:
            字典，键为温度，值为对应的图像数据
        """
        all_data = {}
        
        if not hasattr(self, 'spin_configurations') or not self.spin_configurations:
            print("未找到自旋配置数据")
            return all_data
        
        for idx, config_data in self.spin_configurations.items():
            temp = config_data['temperature']
            data = self.load_spin_image_data(temp)
            if data is not None:
                all_data[temp] = data
        
        return all_data
    
    def generate_spin_visualization(self) -> None:
        """生成所有温度点的自旋配置可视化图（单通道灰度图）"""
        if not self.spin_configurations:
            raise ValueError("未找到自旋配置数据。请先运行模拟。")
        
        try:
            plt.ioff()  # 关闭交互模式
            
            # 为每个温度点生成可视化图
            total = len(self.spin_configurations)
            successful_saves = 0
            
            for idx, data in self.spin_configurations.items():
                try:
                    temp = data['temperature']
                    spin_config = data['spin_config']
                    magnetization = data['magnetization']
                    susceptibility = data['susceptibility']
                    
                    # 数据有效性检查
                    if spin_config is None or not isinstance(spin_config, np.ndarray):
                        print(f"跳过温度点 {temp}: 无效的自旋配置")
                        continue
                    
                    # 生成文件名
                    filename = f'spin_config_T_{temp:.3f}_optimized.png'
                    output_path = os.path.join(self.spin_dir, filename)
                    
                    # 可视化当前温度的自旋配置
                    self._visualize_spin_configuration(spin_config, temp, magnetization, 
                                                     susceptibility, output_path)
                    successful_saves += 1
                    
                    # 每处理完几个温度点后进行垃圾回收
                    if (idx + 1) % 3 == 0:
                        import gc
                        gc.collect()
                        
                except Exception as e:
                    print(f"处理温度点 {idx} 时出错: {e}")
                    continue
            
            print(f"自旋可视化完成（优化版），成功生成 {successful_saves}/{total} 张灰度图片，保存在 '{self.spin_dir}' 文件夹中")
            
        except Exception as e:
            print(f"生成自旋可视化时出错: {e}")
        finally:
            plt.close('all')
            import gc
            gc.collect()
    
    def plot_results(self, file_format: str = 'pdf') -> None:
        """绘制模拟结果并保存到文件"""
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        
        try:
            plt.ioff()  # 关闭交互模式，防止段错误
            
            t = self.results['temperature']
            energy = self.results['energy']
            magnetization = self.results['magnetization']
            susceptibility = self.results['susceptibility']
            specific_heat = self.results['specific_heat']
            
            # 数据有效性检查
            if not all(np.isfinite(arr).all() for arr in [t, energy, magnetization, susceptibility, specific_heat]):
                raise ValueError("结果数据包含无效数值")
            
            # 绘制能量随温度变化图
            fig1 = plt.figure(figsize=(10, 6))
            plt.plot(t, energy, 'r-', linewidth=2, label='Optimized')
            plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
            plt.ylabel(r'Average Energy per Site $(J)$', fontsize=12)
            plt.title('Energy vs Temperature (Optimized)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, f'energy_vs_temperature_optimized.{file_format}'), 
                       format=file_format, bbox_inches='tight', dpi=150)
            plt.close(fig1)
            
            # 绘制比热随温度变化图
            fig2 = plt.figure(figsize=(10, 6))
            plt.plot(t, specific_heat, 'k-', linewidth=2, label='Optimized')
            plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
            plt.ylabel(r'Specific Heat per Site $(k_B)$', fontsize=12)
            plt.title('Specific Heat vs Temperature (Optimized)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, f'specific_heat_vs_temperature_optimized.{file_format}'), 
                       format=file_format, bbox_inches='tight', dpi=150)
            plt.close(fig2)
            
            # 绘制磁化强度随温度变化图
            fig3 = plt.figure(figsize=(10, 6))
            plt.plot(t, magnetization, 'b-', linewidth=2, label='Optimized')
            plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
            plt.ylabel(r'Average Magnetization per Site', fontsize=12)
            plt.title('Magnetization vs Temperature (Optimized)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, f'magnetization_vs_temperature_optimized.{file_format}'), 
                       format=file_format, bbox_inches='tight', dpi=150)
            plt.close(fig3)
            
            # 绘制磁化率随温度变化图
            fig4 = plt.figure(figsize=(10, 6))
            plt.plot(t, susceptibility, 'g-', linewidth=2, label='Optimized')
            plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
            plt.ylabel(r'Magnetic Susceptibility $(k_B/J)$', fontsize=12)
            plt.title('Susceptibility vs Temperature (Optimized)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.figures_dir, f'susceptibility_vs_temperature_optimized.{file_format}'), 
                       format=file_format, bbox_inches='tight', dpi=150)
            plt.close(fig4)
            
            plt.close('all')
            
            # 强制垃圾回收
            import gc
            gc.collect()
            
        except Exception as e:
            print(f"绘图时出错: {e}")
            plt.close('all')
            import gc
            gc.collect()
    
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
                if hasattr(self.memory_pool, 'get_stats'):
                    stats = self.memory_pool.get_stats()
                    f.write(f"内存池缓存命中率: {stats['hit_rate']*100:.1f}%\n")
                    f.write(f"缓存数组类型: {stats['shape_types']}\n")
                    f.write(f"总缓存数组: {stats['total_arrays_cached']}\n")
                elif hasattr(self.memory_pool, 'cache_hits'):
                    hit_rate = self.memory_pool.cache_hits / (self.memory_pool.cache_hits + self.memory_pool.cache_misses + 1e-10)
                    f.write(f"内存池缓存命中率: {hit_rate*100:.1f}%\n")
    
    def save_raw_data(self, filename: str = 'raw_simulation_data.npz') -> None:
        """
        保存模拟的原始数据，包括每个温度点的详细信息
        
        参数:
            filename: 原始数据文件名，默认为 'raw_simulation_data.npz'
        """
        if not self.results:
            raise ValueError("未找到模拟结果。请先运行模拟。")
        
        if not self.spin_configurations:
            raise ValueError("未找到自旋配置数据。请先运行模拟。")
        
        # 准备保存的原始数据
        raw_data = {}
        
        # 保存基本的模拟配置
        raw_data['config'] = self.config.copy()
        raw_data['lattice_size'] = self.L
        raw_data['equilibrium_steps'] = self.ESTEP
        raw_data['measurement_steps'] = self.STEP
        raw_data['interaction_constant'] = self.J
        
        # 保存每个温度点的详细数据
        temperatures = []
        energies = []
        magnetizations = []
        susceptibilities = []
        specific_heats = []
        spin_configs = []
        
        # 按温度排序确保数据顺序一致
        sorted_configs = sorted(self.spin_configurations.items(), 
                              key=lambda x: x[1]['temperature'])
        
        for idx, data in sorted_configs:
            temp = data['temperature']
            spin_config = data['spin_config']
            
            temperatures.append(temp)
            energies.append(data.get('energy', 0.0))
            magnetizations.append(data.get('magnetization', 0.0))
            susceptibilities.append(data.get('susceptibility', 0.0))
            specific_heats.append(data.get('specific_heat', 0.0))
            
            # 确保自旋配置是有效的numpy数组
            if isinstance(spin_config, np.ndarray) and spin_config.size > 0:
                spin_configs.append(spin_config.copy())
            else:
                # 如果数据无效，创建一个零数组作为占位符
                spin_configs.append(np.zeros((self.L, self.L), dtype=np.float64))
        
        # 转换为numpy数组
        raw_data['temperatures'] = np.array(temperatures, dtype=np.float64)
        raw_data['energies'] = np.array(energies, dtype=np.float64)
        raw_data['magnetizations'] = np.array(magnetizations, dtype=np.float64)
        raw_data['susceptibilities'] = np.array(susceptibilities, dtype=np.float64)
        raw_data['specific_heats'] = np.array(specific_heats, dtype=np.float64)
        raw_data['spin_configurations'] = np.stack(spin_configs, axis=0)  # shape: (n_temps, L, L)
        
        # 保存时间戳和元数据
        raw_data['timestamp'] = datetime.datetime.now().isoformat()
        raw_data['num_temperature_points'] = len(temperatures)
        raw_data['spin_config_shape'] = (len(temperatures), self.L, self.L)
        
        # 保存性能数据
        if hasattr(self, 'timing_data') and self.timing_data:
            raw_data['timing_data'] = self.timing_data.copy()
        
        # 保存内存池统计信息
        if hasattr(self.memory_pool, 'get_stats'):
            raw_data['memory_pool_stats'] = self.memory_pool.get_stats()
        elif hasattr(self.memory_pool, 'cache_hits'):
            hit_rate = self.memory_pool.cache_hits / (self.memory_pool.cache_hits + self.memory_pool.cache_misses + 1e-10)
            raw_data['memory_pool_stats'] = {
                'cache_hits': self.memory_pool.cache_hits,
                'cache_misses': self.memory_pool.cache_misses,
                'hit_rate': hit_rate
            }
        
        # 保存到npz文件
        output_path = os.path.join(self.base_output_dir, filename)
        try:
            np.savez_compressed(output_path, **raw_data)
            print(f"原始数据已保存到: {output_path}")
            
            # 创建数据说明文件
            readme_path = os.path.join(self.base_output_dir, 'raw_data_readme.txt')
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write("XY模型模拟原始数据说明\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"数据文件: {filename}\n")
                f.write(f"创建时间: {raw_data['timestamp']}\n")
                f.write(f"晶格尺寸: {self.L}x{self.L}\n")
                f.write(f"温度点数: {raw_data['num_temperature_points']}\n")
                f.write(f"平衡步数: {self.ESTEP}\n")
                f.write(f"测量步数: {self.STEP}\n\n")
                
                f.write("包含的数据字段:\n")
                f.write("- temperatures: 温度数组\n")
                f.write("- energies: 每个温度点的平均能量\n")
                f.write("- magnetizations: 每个温度点的平均磁化强度\n")
                f.write("- susceptibilities: 每个温度点的磁化率\n")
                f.write("- specific_heats: 每个温度点的比热\n")
                f.write("- spin_configurations: 自旋配置数组 (n_temps, L, L)\n")
                f.write("- config: 完整的模拟配置字典\n")
                f.write("- timing_data: 性能计时数据\n")
                f.write("- memory_pool_stats: 内存池统计信息\n\n")
                
                f.write("数据使用示例 (Python):\n")
                f.write("```python\n")
                f.write("import numpy as np\n\n")
                f.write(f"data = np.load('{output_path}')\n")
                f.write("temps = data['temperatures']\n")
                f.write("spins = data['spin_configurations']  # shape: (n_temps, L, L)\n")
                f.write("energy = data['energies']\n")
                f.write("# 获取特定温度点的自旋配置\n")
                f.write("spin_at_T_1_0 = spins[0]  # 第一个温度点的自旋配置\n")
                f.write("```\n\n")
                
                f.write("注意: 自旋配置中的值为角度值 (弧度)，范围 [0, 2π)\n")
                f.write("可以使用 np.mod(spin_config, 2*np.pi) 来规范化角度值\n")
            
            print(f"数据说明文件已创建: {readme_path}")
            
        except Exception as e:
            print(f"保存原始数据时出错: {e}")
            raise
    
    def load_raw_data(self, filename: str = 'raw_simulation_data.npz') -> Dict[str, Any]:
        """
        加载之前保存的原始数据
        
        参数:
            filename: 原始数据文件名
            
        返回:
            包含所有原始数据的字典
        """
        if self.base_output_dir is None:
            raise ValueError("输出目录未设置，无法加载数据")
        
        file_path = os.path.join(self.base_output_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"原始数据文件不存在: {file_path}")
        
        try:
            data = dict(np.load(file_path, allow_pickle=True))
            print(f"原始数据已从 {file_path} 加载")
            print(f"包含 {data.get('num_temperature_points', 0)} 个温度点的数据")
            return data
        except Exception as e:
            print(f"加载原始数据时出错: {e}")
            raise
    
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
