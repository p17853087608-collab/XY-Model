"""
XY Model Simulator using Swendsen-Wang Algorithm with GPU Acceleration

This module provides an optimized class for simulating 2D XY model using 
Swendsen-Wang clustering algorithm with optional GPU acceleration.
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

# 尝试导入CuPy进行GPU加速
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    gpu_available = False


class XYModelSimulator:
    """
    XY模型模拟器类：使用Swendsen-Wang算法进行蒙特卡洛模拟（GPU加速版）
    
    该模拟器支持CPU和GPU加速计算，能够高效计算不同温度下的物理量，如能量、磁化强度、比热和磁化率等。
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
        # 确定是否使用GPU
        self.use_gpu = use_gpu and gpu_available
        self.device = 'GPU' if self.use_gpu else 'CPU'
        self.xp = cp if self.use_gpu else np
        
        self.L = lattice_size
        self.ESTEP = equilibrium_steps
        self.STEP = measurement_steps
        self.J = interaction_constant
        
        # 设置随机种子
        if random_seed is not None:
            if self.use_gpu:
                cp.random.seed(random_seed)
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
        
        # 预计算周期性边界索引数组
        self._setup_boundary_arrays()
        
        # 生成唯一的结果文件夹名称（基于时间戳）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_dir = f"simulation_results_{timestamp}"
        
        # 创建主结果文件夹
        os.makedirs(self.base_output_dir, exist_ok=True)
        
        # 创建子文件夹
        self.figures_dir = os.path.join(self.base_output_dir, "figures")
        self.spin_dir = os.path.join(self.base_output_dir, "spin_configurations")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.spin_dir, exist_ok=True)
        
        print(f"XY模型模拟器初始化完成。计算设备: {self.device}，晶格尺寸: {self.L}x{self.L}")
    
    def _setup_boundary_arrays(self):
        """预计算周期性边界索引数组以提高性能"""
        # 创建CPU上的索引数组，然后根据需要转移到GPU
        i_prev_cpu = np.arange(self.L) - 1
        i_next_cpu = np.arange(self.L) + 1
        i_next_cpu[-1] = 0  # 周期性边界
        
        j_prev_cpu = np.arange(self.L) - 1
        j_next_cpu = np.arange(self.L) + 1
        j_next_cpu[-1] = 0  # 周期性边界
        
        # 如果使用GPU，将索引数组转移到GPU
        if self.use_gpu:
            self.i_prev = cp.asarray(i_prev_cpu)
            self.i_next = cp.asarray(i_next_cpu)
            self.j_prev = cp.asarray(j_prev_cpu)
            self.j_next = cp.asarray(j_next_cpu)
        else:
            self.i_prev = i_prev_cpu
            self.i_next = i_next_cpu
            self.j_prev = j_prev_cpu
            self.j_next = j_next_cpu
    
    def _initialize_spins(self) -> Any:
        """
        初始化XY模型的自旋角度
        
        返回:
            自旋角度数组（CPU或GPU数组）
        """
        if self.use_gpu:
            return cp.random.rand(self.L, self.L) * 2 * np.pi
        else:
            return np.random.rand(self.L, self.L) * 2 * np.pi
    
    def _to_cpu(self, array: Any) -> np.ndarray:
        """将数组转移到CPU（如果在GPU上）"""
        if self.use_gpu and isinstance(array, cp.ndarray):
            return cp.asnumpy(array)
        return array
    
    def _to_gpu(self, array: Any) -> Any:
        """将数组转移到GPU（如果可用）"""
        if self.use_gpu and isinstance(array, np.ndarray):
            return cp.asarray(array)
        return array
    
    def _freeze_bonds(self, ising: Any, temperature: float, 
                      s_matrix: Any) -> Tuple[Any, Any]:
        """
        根据Swendsen-Wang算法概率性地冻结相同自旋方向的键
        
        参数:
            ising: Ising自旋配置
            temperature: 系统温度
            s_matrix: 自旋幅度矩阵
            
        返回:
            水平和垂直方向的冻结键矩阵
        """
        i_bond_frozen = self.xp.zeros([self.L, self.L], dtype=bool)
        j_bond_frozen = self.xp.zeros([self.L, self.L], dtype=bool)
        
        # 计算与右侧邻居形成冻结键的概率
        freeze_prob_nexti = 1 - self.xp.exp(-2 * self.J * s_matrix * s_matrix[self.i_next] / temperature)
        # 计算与上方邻居形成冻结键的概率
        freeze_prob_nextj = 1 - self.xp.exp(-2 * self.J * s_matrix * s_matrix[:, self.j_next] / temperature)
        
        # 生成随机数
        rand_vals_i = self.xp.random.rand(self.L, self.L)
        rand_vals_j = self.xp.random.rand(self.L, self.L)
        
        # 确定冻结键
        i_bond_frozen = (ising == ising[self.i_next]) & (rand_vals_i < freeze_prob_nexti)
        j_bond_frozen = (ising == ising[:, self.j_next]) & (rand_vals_j < freeze_prob_nextj)
        
        return i_bond_frozen, j_bond_frozen
    
    def _proper_label(self, prp_label: Any, i: int) -> int:
        """
        H-K算法（Hoshen-Kopelman）用于正确标记聚类标签
        
        参数:
            prp_label: 标签数组
            i: 标签索引
            
        返回:
            聚类的根标签
        """
        # 确保在CPU上执行此操作，因为它涉及复杂的索引操作
        if self.use_gpu:
            prp_label_cpu = cp.asnumpy(prp_label)
            i_cpu = int(i)
            
            if i_cpu < 0 or i_cpu >= len(prp_label_cpu):
                return 0
            
            # 查找根标签
            while prp_label_cpu[i_cpu] != i_cpu:
                i_cpu = prp_label_cpu[i_cpu]
                if i_cpu < 0 or i_cpu >= len(prp_label_cpu):
                    return 0
            
            return i_cpu
        else:
            if i < 0 or i >= len(prp_label):
                return 0
            
            # 查找根标签
            while prp_label[i] != i:
                i = prp_label[i]
                if i < 0 or i >= len(prp_label):
                    return 0
            
            return i
    
    def _cluster_find(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """
        Swendsen-Wang聚类查找：根据冻结键识别自旋聚类
        
        参数:
            i_bond_frozen: 水平方向的冻结键
            j_bond_frozen: 垂直方向的冻结键
            
        返回:
            聚类矩阵和标签数组
        """
        cluster = self.xp.zeros([self.L, self.L], dtype=int)
        prp_label = self.xp.arange(self.L**2, dtype=int)  # 初始时每个标签指向自身
        current_label = 0
        
        # 转换为CPU进行聚类查找（复杂的条件操作在CPU上通常更快）
        i_bond_frozen_cpu = self._to_cpu(i_bond_frozen)
        j_bond_frozen_cpu = self._to_cpu(j_bond_frozen)
        cluster_cpu = np.zeros([self.L, self.L], dtype=int)
        prp_label_cpu = np.arange(self.L**2, dtype=int)
        
        for i in range(self.L):
            for j in range(self.L):
                bonds = 0
                ibonds = []
                jbonds = []
                
                # 检查与左侧格点(i-1,j)的冻结键
                if i > 0 and i_bond_frozen_cpu[i-1, j]:
                    ibonds.append(i-1)
                    jbonds.append(j)
                    bonds += 1
                # 检查与右侧格点(i+1,j)的冻结键（考虑周期性边界）
                if i == self.L-1 and i_bond_frozen_cpu[i, j]:
                    ibonds.append(0)
                    jbonds.append(j)
                    bonds += 1
                # 检查与下方格点(i,j-1)的冻结键
                if j > 0 and j_bond_frozen_cpu[i, j-1]:
                    ibonds.append(i)
                    jbonds.append(j-1)
                    bonds += 1
                # 检查与上方格点(i,j+1)的冻结键（考虑周期性边界）
                if j == self.L-1 and j_bond_frozen_cpu[i, j]:
                    ibonds.append(i)
                    jbonds.append(0)
                    bonds += 1
                
                # 如果没有连接的冻结键，创建新聚类
                if bonds == 0:
                    cluster_cpu[i, j] = current_label
                    prp_label_cpu[current_label] = current_label
                    current_label += 1
                # 有连接的冻结键，进行聚类合并
                else:
                    min_label = current_label
                    # 查找相邻格点中的最小聚类标签
                    for b in range(bonds):
                        plabel = self._proper_label(prp_label_cpu, cluster_cpu[ibonds[b], jbonds[b]])
                        if min_label > plabel:
                            min_label = plabel
                    
                    cluster_cpu[i, j] = min_label
                    # 链接所有相邻聚类的标签到最小标签
                    for b in range(bonds):
                        plabel_n = cluster_cpu[ibonds[b], jbonds[b]]
                        prp_label_cpu[plabel_n] = min_label
                        cluster_cpu[ibonds[b], jbonds[b]] = min_label
        
        # 将结果转回GPU（如果使用GPU）
        cluster = self._to_gpu(cluster_cpu)
        prp_label = self._to_gpu(prp_label_cpu)
        
        return cluster, prp_label
    
    def _flip_cluster(self, ising: Any, cluster: Any, 
                      prp_label: Any) -> Tuple[Any, int]:
        """
        以0.5概率翻转整个聚类的自旋方向
        
        参数:
            ising: Ising自旋配置
            cluster: 聚类矩阵
            prp_label: 标签数组
            
        返回:
            更新后的自旋配置和翻转的自旋数量
        """
        # 转换为CPU进行聚类翻转操作
        cluster_cpu = self._to_cpu(cluster).copy()
        prp_label_cpu = self._to_cpu(prp_label)
        ising_cpu = self._to_cpu(ising).copy()
        
        # 重新标记所有聚类标签为正确的根标签
        for i in range(self.L):
            for j in range(self.L):
                cluster_cpu[i, j] = self._proper_label(prp_label_cpu, cluster_cpu[i, j])
        
        # 决定每个聚类的翻转情况
        num_clusters = int(cluster_cpu.max() + 1)
        flip_decision = np.random.rand(num_clusters) < 0.5
        
        # 应用翻转
        flips = 0
        for i in range(self.L):
            for j in range(self.L):
                label = cluster_cpu[i, j]
                if flip_decision[label]:
                    ising_cpu[i, j] *= -1
                    flips += 1
        
        # 将结果转回GPU（如果使用GPU）
        ising = self._to_gpu(ising_cpu)
        
        return ising, flips
    
    def _one_mc_step_ising(self, ising: Any, s_matrix: Any, 
                          temperature: float) -> Any:
        """
        Ising模型的Swendsen-Wang算法单步更新
        
        参数:
            ising: Ising自旋配置
            s_matrix: 自旋幅度矩阵
            temperature: 系统温度
            
        返回:
            更新后的自旋配置
        """
        # 冻结键
        i_bond_frozen, j_bond_frozen = self._freeze_bonds(ising, temperature, s_matrix)
        
        # 聚类查找
        cluster, prp_label = self._cluster_find(i_bond_frozen, j_bond_frozen)
        
        # 聚类翻转
        ising, _ = self._flip_cluster(ising, cluster, prp_label)
        
        return ising
    
    def _decompose_xy(self, xy: Any, proj: float) -> Tuple[Any, Any, Any, Any]:
        """
        将XY模型分解为两个Ising模型（沿投影方向）
        
        参数:
            xy: XY自旋配置
            proj: 投影方向角度
            
        返回:
            Ising x、Ising y、S_x幅度和S_y幅度
        """
        cos_xy = self.xp.cos(xy)
        sin_xy = self.xp.sin(xy)
        
        cos_proj = np.cos(proj)
        sin_proj = np.sin(proj)
        
        # 旋转坐标系到投影方向
        x_rot = cos_xy * cos_proj + sin_xy * sin_proj
        y_rot = -cos_xy * sin_proj + sin_xy * cos_proj
        
        ising_x = self.xp.sign(x_rot)
        ising_y = self.xp.sign(y_rot)
        s_x = self.xp.abs(x_rot)
        s_y = self.xp.abs(y_rot)
        
        return ising_x, ising_y, s_x, s_y
    
    def _compose_xy(self, ising_x_new: Any, ising_y_new: Any, 
                   proj: float, s_x: Any, s_y: Any) -> Any:
        """
        将两个Ising模型组合回XY模型
        
        参数:
            ising_x_new: 更新后的Ising x配置
            ising_y_new: 更新后的Ising y配置
            proj: 投影方向角度
            s_x: x方向幅度
            s_y: y方向幅度
            
        返回:
            组合后的XY自旋配置
        """
        cos_proj = np.cos(proj)
        sin_proj = np.sin(proj)
        
        # 旋转回原始坐标系
        x_rot_new = ising_x_new * s_x
        y_rot_new = ising_y_new * s_y
        x_new = x_rot_new * cos_proj - y_rot_new * sin_proj
        y_new = x_rot_new * sin_proj + y_rot_new * cos_proj
        
        xy_new = self.xp.arctan2(y_new, x_new)
        
        return xy_new
    
    def _one_mc_step_xy(self, xy: Any, temperature: float) -> Any:
        """
        XY模型的单步蒙特卡洛更新
        
        参数:
            xy: XY自旋配置
            temperature: 系统温度
            
        返回:
            更新后的XY自旋配置
        """
        # 随机选择投影方向
        proj = np.random.rand() * 2 * np.pi
        
        # 分解XY模型
        ising_x, ising_y, s_x, s_y = self._decompose_xy(xy, proj)
        
        # 更新x分量
        ising_x_new = self._one_mc_step_ising(ising_x, s_x, temperature)
        
        # 更新y分量
        ising_y_new = self._one_mc_step_ising(ising_y, s_y, temperature)
        
        # 组合回XY模型
        xy_new = self._compose_xy(ising_x_new, ising_y_new, proj, s_x, s_y)
        
        return xy_new
    
    def _calculate_energy_magnetization(self, xy: Any) -> Tuple[float, float]:
        """
        计算XY模型的能量和磁化强度
        
        参数:
            xy: XY自旋配置
            
        返回:
            能量和磁化强度
        """
        # 计算能量（在GPU上执行）
        cos_xy = self.xp.cos(xy)
        sin_xy = self.xp.sin(xy)
        
        energy = 0.0
        # 使用向量化操作计算能量
        energy -= self.xp.sum(cos_xy * cos_xy[self.i_next] + sin_xy * sin_xy[self.i_next])  # 水平邻居
        energy -= self.xp.sum(cos_xy * cos_xy[:, self.j_next] + sin_xy * sin_xy[:, self.j_next])  # 垂直邻居
        
        # 转换为标量并应用因子
        energy = float(energy) * 0.5
        
        # 计算磁化强度
        mag_x = float(self.xp.sum(cos_xy))
        mag_y = float(self.xp.sum(sin_xy))
        magnetization = np.sqrt(mag_x**2 + mag_y**2) / (self.L**2)
        
        return energy, magnetization
    
    def run_simulation(self, temperature_range: Tuple[float, float] = (0.1, 2.5), 
                      num_temperatures: int = 10) -> Dict[str, Any]:
        """
        在温度范围内运行XY模型模拟
        
        参数:
            temperature_range: (T_min, T_max)温度范围，默认值: (0.1, 2.5)
            num_temperatures: 温度点数量，默认值: 10
            
        返回:
            包含温度、能量、磁化强度等物理量的模拟结果字典
        """
        start_time = time.time()
        
        t_min, t_max = temperature_range
        temperature_array = np.linspace(t_min, t_max, num_temperatures)
        
        magnetization_array = np.zeros(num_temperatures)
        energy_array = np.zeros(num_temperatures)
        susceptibility_array = np.zeros(num_temperatures)
        specific_heat_array = np.zeros(num_temperatures)
        
        # 存储每个温度点计算时间
        temperature_times = np.zeros(num_temperatures)
        
        # 遍历各温度点进行模拟
        for idx, temp in enumerate(temperature_array):
            temp_start = time.time()
            
            # 初始化自旋
            xy = self._initialize_spins()
            
            # 热化过程：让系统达到平衡状态
            for _ in range(self.ESTEP):
                xy = self._one_mc_step_xy(xy, temp)
            
            # 测量过程：计算物理量的统计平均值
            energy_sum = 0.0
            magnetization_sum = 0.0
            energy_sq_sum = 0.0
            magnetization_sq_sum = 0.0
            
            for _ in range(self.STEP):
                xy = self._one_mc_step_xy(xy, temp)
                energy, magnetization = self._calculate_energy_magnetization(xy)
                
                energy_sum += energy
                magnetization_sum += magnetization
                energy_sq_sum += energy**2
                magnetization_sq_sum += magnetization**2
            
            # 计算平均值
            energy_mean = energy_sum / self.STEP / (self.L**2)  # 每格点平均能量
            magnetization_mean = magnetization_sum / self.STEP  # 平均磁化强度
            energy_sq_mean = energy_sq_sum / self.STEP / (self.L**4)  # 能量平方的平均值
            magnetization_sq_mean = magnetization_sq_sum / self.STEP  # 磁化强度平方的平均值
            
            # 计算派生物理量
            susceptibility = (magnetization_sq_mean - magnetization_mean**2) / temp
            specific_heat = (energy_sq_mean - energy_mean**2) / (temp**2)
            
            # 存储结果
            energy_array[idx] = energy_mean
            magnetization_array[idx] = magnetization_mean
            susceptibility_array[idx] = susceptibility
            specific_heat_array[idx] = specific_heat
            
            # 保存自旋配置（转移到CPU）
            xy_cpu = self._to_cpu(xy)
            self.spin_configurations[idx] = {
                'temperature': temp,
                'spin_config': xy_cpu,
                'magnetization': magnetization_mean,
                'susceptibility': susceptibility
            }
            
            # 记录时间
            temperature_times[idx] = time.time() - temp_start
            
        # 总模拟时间
        total_time = time.time() - start_time
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times.mean()
        }
        
        # 存储结果
        self.results = {
            'temperature': temperature_array,
            'energy': energy_array,
            'magnetization': magnetization_array,
            'specific_heat': specific_heat_array,
            'susceptibility': susceptibility_array,
            'config': self.config.copy(),
            'timing': self.timing_data
        }
        
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
        # 使用自旋角度映射到HSV色彩空间，然后转换为RGB
        # 将角度从[0, 2π)映射到[0, 1)的色调值，每0.5度一个色彩
        hue = (spin_config % (2 * np.pi)) / (2 * np.pi)
        
        # 使用磁化强度作为饱和度（归一化到[0.3, 1.0]）
        saturation = 0.3 + 0.7 * magnetization
        
        # 使用恒定的高亮度值
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
        title_text = f'XY Model Spin Configuration (Temperature = {temperature:.3f})'
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
            filename = f'spin_config_T_{temp:.3f}.png'
            output_path = os.path.join(self.spin_dir, filename)
            
            # 可视化当前温度的自旋配置
            self._visualize_spin_configuration(spin_config, temp, magnetization, 
                                             susceptibility, output_path)
        
        print(f"自旋可视化完成，共生成 {total} 张彩色图片，保存在 '{self.spin_dir}' 文件夹中")
    
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
        plt.plot(t, energy, 'rx-', linewidth=2)
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Energy per Site $(J)$', fontsize=12)
        plt.title('Energy vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, f'energy_vs_temperature.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制比热随温度变化图
        plt.figure()
        plt.plot(t, specific_heat, 'kx-', linewidth=2)
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Specific Heat per Site $(k_B)$', fontsize=12)
        plt.title('Specific Heat vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, f'specific_heat_vs_temperature.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化强度随温度变化图
        plt.figure()
        plt.plot(t, magnetization, 'bx-', linewidth=2)
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Average Magnetization per Site', fontsize=12)
        plt.title('Magnetization vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, f'magnetization_vs_temperature.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        # 绘制磁化率随温度变化图
        plt.figure()
        plt.plot(t, susceptibility, 'gx-', linewidth=2)
        plt.xlabel(r'Temperature $(k_BT/J)$', fontsize=12)
        plt.ylabel(r'Magnetic Susceptibility $(k_B/J)$', fontsize=12)
        plt.title('Susceptibility vs Temperature', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.figures_dir, f'susceptibility_vs_temperature.{file_format}'), 
                   format=file_format, bbox_inches='tight', dpi=300)
        
        plt.close('all')
    
    def save_results(self, filename: str = 'simulation_results.txt') -> None:
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
        header = "# Temperature\tEnergy\tSpecific Heat\tMagnetization\tSusceptibility"
        np.savetxt(output_path, data, header=header, delimiter='\t', fmt='%.6f')
        
        # 保存配置信息
        config_path = os.path.join(self.base_output_dir, 'simulation_config.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("XY Model Simulation Configuration\n")
            f.write("================================\n")
            for key, value in self.results['config'].items():
                f.write(f"{key}: {value}\n")
            
            if 'timing' in self.results:
                f.write("\nPerformance Data\n")
                f.write("===============\n")
                f.write(f"总运行时间: {self.results['timing']['total_time']:.2f}秒\n")
                f.write(f"平均每个温度点时间: {self.results['timing']['per_temperature_time']:.2f}秒\n")
    
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