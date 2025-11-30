    def _setup_boundary_arrays(self):
        """预计算周期性边界索引数组以提高性能"""
        # 优化1: 使用单个数组创建，减少重复计算
        indices = np.arange(self.L)
        
        # 优化2: 向量化操作，避免重复边界设置
        # 使用模运算实现周期性边界，避免条件分支
        i_prev_cpu = (indices - 1) % self.L
        i_next_cpu = (indices + 1) % self.L
        j_prev_cpu = (indices - 1) % self.L
        j_next_cpu = (indices + 1) % self.L
        
        # 优化3: 批量分配，减少函数调用开销
        if self.use_gpu:
            # 一次性转换所有数组到GPU，减少GPU-CPU通信
            gpu_arrays = cp.asarray([i_prev_cpu, i_next_cpu, j_prev_cpu, j_next_cpu])
            self.i_prev, self.i_next, self.j_prev, self.j_next = gpu_arrays
        else:
            # 直接引用，避免不必要的内存拷贝
            self.i_prev = i_prev_cpu
            self.i_next = i_next_cpu
            self.j_prev = j_prev_cpu
            self.j_next = j_next_cpu