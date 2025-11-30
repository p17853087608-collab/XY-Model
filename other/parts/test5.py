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
        
        # 确保输入数据形状正确
        if cluster_cpu.size == 0 or cluster_cpu.ndim != 2:
            return self._to_gpu(ising_cpu), 0
        
        # 优化1: 预先计算数组长度，避免重复调用
        array_size = cluster_cpu.size
        cluster_shape = cluster_cpu.shape
        
        # 优化2: 使用更高效的向量化重新标记
        # 首先检查并扩展prp_label数组（如果需要）
        max_cluster_val = cluster_cpu.max()
        if prp_label_cpu.size <= max_cluster_val:
            new_size = max(max_cluster_val + 1, prp_label_cpu.size * 2)  # 优化：使用2倍扩展策略
            prp_label_extended = np.arange(new_size, dtype=prp_label_cpu.dtype)
            prp_label_extended[:prp_label_cpu.size] = prp_label_cpu
            prp_label_cpu = prp_label_extended
        
        # 优化3: 使用字典缓存已计算的根标签，避免重复计算
        root_cache = {}
        
        def cached_proper_label(label):
            if label not in root_cache:
                root_cache[label] = self._proper_label(prp_label_cpu, label)
            return root_cache[label]
        
        # 向量化重新标记，使用缓存的函数
        vectorized_relabel = np.vectorize(cached_proper_label)
        cluster_cpu = vectorized_relabel(cluster_cpu)
        
        # 优化4: 使用更高效的聚类翻转决策
        cluster_max = cluster_cpu.max()
        if cluster_max < 0:
            return self._to_gpu(ising_cpu), 0
            
        num_clusters = int(cluster_max + 1)
        
        # 优化5: 预先生成随机数，避免重复调用random.rand
        flip_decision = np.random.random(num_clusters) < 0.5
        
        # 优化6: 使用高效的向量化翻转操作
        # 首先创建翻转掩码
        cluster_masked = np.clip(cluster_cpu, 0, num_clusters - 1)
        flip_mask = flip_decision[cluster_masked]
        
        # 优化7: 使用np.where进行条件翻转，避免循环
        ising_cpu = np.where(flip_mask, -ising_cpu, ising_cpu)
        
        # 计算翻转的自旋数量
        flips = int(np.sum(flip_mask))
        
        # 将结果转回GPU（如果使用GPU）
        ising = self._to_gpu(ising_cpu)
        
        return ising, flips