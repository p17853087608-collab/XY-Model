"""
XY模型模拟器 - 核心模拟引擎
使用Swendsen-Wang算法进行蒙特卡洛模拟，包含核心模拟逻辑
"""

import numpy as np
from typing import Tuple, Dict, Any
from functools import lru_cache
from xy_utils import UltraSmartMemoryPool, AdvancedTrigTable, OptimizedUnionFind


class XYModelSimulatorCore:
    """XY模型核心模拟器类"""
    
    def __init__(self, lattice_size: int = 16, equilibrium_steps: int = 1000, 
                 measurement_steps: int = 10000, interaction_constant: float = 1.0,
                 random_seed: int = None, process_id: int = 0):
        """
        初始化XY模型核心模拟器
        
        参数:
            lattice_size: 晶格尺寸(LxL)，默认值: 16
            equilibrium_steps: 系统平衡步数，默认值: 1000
            measurement_steps: 物理量测量步数，默认值: 10000
            interaction_constant: 交换相互作用常数J，默认值: 1.0
            random_seed: 随机种子，默认值: None
            process_id: 进程ID
        """
        self.L = lattice_size
        self.lattice_size = lattice_size
        self.ESTEP = equilibrium_steps
        self.STEP = measurement_steps
        self.J = interaction_constant
        self.process_id = process_id
        self.xp = np
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            self.process_seed = random_seed
        else:
            import time
            self.process_seed = int(time.time()) % 1000000
        
        # 性能优化：预计算周期性边界索引数组
        self._setup_boundary_arrays()
        
        # 性能优化：初始化高级内存池
        pool_size = max(15, min(25, self.L // 2))
        self.memory_pool = UltraSmartMemoryPool(self.xp, max_arrays_per_shape=pool_size, process_id=process_id)
        
        # 性能优化：预计算常量
        self.two_pi = 2.0 * np.pi
        self.inv_L = 1.0 / self.L
        self.l_squared = self.L ** 2
        self.inv_l_squared = 1.0 / self.l_squared
        
        # 性能优化：初始化高级三角函数查找表
        self.trig_table = AdvancedTrigTable(resolution=20000)
        
        # 初始化超优化缓存
        self._proj_cache = []
        self._proj_trig_cache = {}
        self._freeze_cache = {}
        self._cluster_cache = {}
    
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
    
    @lru_cache(maxsize=256)
    def _get_trig_values(self, angle_float: float) -> Tuple[float, float]:
        """缓存三角函数计算结果"""
        angle_rad = float(angle_float)
        return np.cos(angle_rad), np.sin(angle_rad)
    
    def _initialize_spins(self) -> Any:
        """初始化XY模型的自旋角度"""
        import random
        
        # 添加额外的随机扰动，确保每次初始化都不同
        if hasattr(self, 'process_seed') and self.process_seed is not None:
            import time
            time_based_seed = (int(time.time() * 1000000) % 1000000)
            task_seed = getattr(self, 'task_id', 0) % 1000000
            
            # 为每次初始化生成独特的随机种子
            init_seed = (self.process_seed + time_based_seed + task_seed) % 1000000
            np.random.seed(init_seed)
        
        # 生成随机自旋配置
        random_vals = self.xp.random.rand(self.L, self.L)
        
        # 添加额外的随机扰动
        if hasattr(self, 'process_seed') and self.process_seed is not None:
            noise_level = 0.01  # 1%的随机噪声
            noise = self.xp.random.rand(self.L, self.L) * noise_level
            random_vals = (random_vals + noise) % 1.0
        
        spin_array = random_vals * self.two_pi
        return spin_array.copy()
    
    def _freeze_bonds_ultra_vectorized(self, ising: Any, temperature: float, 
                                       s_matrix: Any) -> Tuple[Any, Any]:
        """超优化冻结键计算"""
        cache_key = (round(temperature, 8), self.L)
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
        
        # 批量计算冻结概率
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
        elif total_bonds < self.L * self.L * 0.05:
            return self._cluster_find_optimized_union_find(i_bond_frozen, j_bond_frozen)
        elif self.L <= 64:
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
    
    def run_temperature_simulation(self, temperature: float) -> Dict[str, Any]:
        """运行单个温度点的模拟（无收敛检测）"""
        try:
            xy = self._initialize_spins()
            
            # === 热化阶段 ===
            for step in range(self.ESTEP):
                xy = self._one_mc_step_xy_ultra_optimized(xy, temperature)
            
            # === 测量阶段 ===
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
            
            # === 统计分析 ===
            valid_energy = energy_samples[:actual_measurements]
            valid_magnetization = mag_samples[:actual_measurements]
            
            if len(valid_energy) == 0:
                energy_mean = 0.0
                magnetization_mean = 0.0
                susceptibility = 0.0
                specific_heat = 0.0
            else:
                # 计算均值
                try:
                    energy_mean = float(np.mean(valid_energy))
                    magnetization_mean = float(np.mean(valid_magnetization))
                except:
                    energy_mean = 0.0
                    magnetization_mean = 0.0
                
                # 检查均值有效性
                if not (np.isfinite(energy_mean) and np.isfinite(magnetization_mean)):
                    energy_mean = 0.0
                    magnetization_mean = 0.0
                
                # 计算平方均值
                if len(valid_energy) >= 10:
                    try:
                        energy_sq_mean = float(np.mean(valid_energy * valid_energy))
                        mag_sq_mean = float(np.mean(valid_magnetization * valid_magnetization))
                    except:
                        energy_sq_mean = energy_mean ** 2
                        mag_sq_mean = magnetization_mean ** 2
                else:
                    energy_sq_mean = energy_mean ** 2
                    mag_sq_mean = magnetization_mean ** 2
                
                # 计算susceptibility和specific_heat
                try:
                    inv_temp = 1.0 / max(temperature, 1e-10)
                    inv_temp_squared = inv_temp * inv_temp
                    susceptibility = max(0.0, (mag_sq_mean - magnetization_mean * magnetization_mean) * inv_temp * self.l_squared)
                    specific_heat = max(0.0, (energy_sq_mean - energy_mean * energy_mean) * inv_temp_squared * self.l_squared)
                except:
                    susceptibility = 0.0
                    specific_heat = 0.0
            
            if xy is None or not isinstance(xy, np.ndarray):
                xy = self._initialize_spins()
            
            # 内存优化清理
            del valid_energy, valid_magnetization
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
                'actual_equilibrium_steps': self.ESTEP,
                'actual_measurements': actual_measurements
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
