"""
XY模型模拟器 - 工具模块
包含内存池、三角函数表、并查集等工具类
"""

import numpy as np
from typing import Tuple, Any
import time


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
                all(s <= stored_s * 1.1 for s, stored_s in zip(shape, stored_shape))):
                
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
    
    def get_stats(self) -> dict:
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
    
    def __init__(self, resolution: int = 20000):
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
