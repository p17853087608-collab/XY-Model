    def _initialize_spins(self) -> Any:
        """
        初始化XY模型的自旋角度
        
        返回:
            自旋角度数组（CPU或GPU数组）
        """
        # 优化1: 预计算常量，避免重复计算
        two_pi = 2 * np.pi
        
        # 优化2: 使用统一接口创建随机数，减少条件判断
        random_vals = self.xp.random.rand(self.L, self.L)
        
        return random_vals * two_pi
    
    def _to_cpu(self, array: Any) -> np.ndarray:
        """将数组转移到CPU（如果在GPU上）"""
        # 优化3: 缓存类型检查结果，避免重复isinstance调用
        if hasattr(self, '_gpu_array_type'):
            array_type = self._gpu_array_type
        else:
            array_type = cp.ndarray if self.use_gpu else type(None)
            self._gpu_array_type = array_type
        
        if self.use_gpu and isinstance(array, array_type):
            return cp.asnumpy(array)
        return array
    
    def _to_gpu(self, array: Any) -> Any:
        """将数组转移到GPU（如果可用）"""
        # 优化4: 使用缓存的类型检查
        if hasattr(self, '_gpu_array_type'):
            array_type = self._gpu_array_type
        else:
            array_type = cp.ndarray if self.use_gpu else type(None)
            self._gpu_array_type = array_type
        
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
        # 优化5: 预计算指数函数的参数，减少重复计算
        exp_factor = -2 * self.J / temperature
        
        # 优化6: 批量计算所有概率，利用向量化操作
        # 使用预计算的边界索引，避免索引重复计算
        s_next_i = s_matrix[self.i_next]
        s_next_j = s_matrix[:, self.j_next]
        
        freeze_prob_nexti = 1 - self.xp.exp(exp_factor * s_matrix * s_next_i)
        freeze_prob_nextj = 1 - self.xp.exp(exp_factor * s_matrix * s_next_j)
        
        # 优化7: 一次性生成所有随机数，减少函数调用开销
        rand_vals = self.xp.random.rand(self.L, self.L, 2)  # [i,j,方向]
        rand_vals_i = rand_vals[:, :, 0]
        rand_vals_j = rand_vals[:, :, 1]
        
        # 优化8: 使用向量化比较，一次性计算所有冻结键
        # 避免中间变量的内存分配
        i_bond_frozen = (ising == ising[self.i_next]) & (rand_vals_i < freeze_prob_nexti)
        j_bond_frozen = (ising == ising[:, self.j_next]) & (rand_vals_j < freeze_prob_nextj)
        
        return i_bond_frozen, j_bond_frozen