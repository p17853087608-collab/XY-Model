def _cluster_find(self, i_bond_frozen: Any, j_bond_frozen: Any) -> Tuple[Any, Any]:
        """
        Swendsen-Wang聚类查找：根据冻结键识别自旋聚类
        
        参数:
            i_bond_frozen: 水平方向的冻结键
            j_bond_frozen: 垂直方向的冻结键
            
        返回:
            聚类矩阵和标签数组
        """
        # 优化1: 移除不必要的GPU初始化，直接使用CPU版本
        i_bond_frozen_cpu = self._to_cpu(i_bond_frozen)
        j_bond_frozen_cpu = self._to_cpu(j_bond_frozen)
        cluster_cpu = np.zeros([self.L, self.L], dtype=int)
        prp_label_cpu = np.arange(self.L**2, dtype=int)
        
        current_label = 0
        array_len = self.L
        
        for i in range(array_len):
            for j in range(array_len):
                neighbor_indices = []
                neighbor_labels = []
                
                # 优化2: 使用更高效的条件检查，减少重复的边界检查
                # 左侧邻居
                if i > 0 and i_bond_frozen_cpu[i-1, j]:
                    neighbor_indices.append((i-1, j))
                    neighbor_labels.append(cluster_cpu[i-1, j])
                # 右侧邻居（周期性边界）
                elif i == array_len-1 and i_bond_frozen_cpu[i, j]:
                    neighbor_indices.append((0, j))
                    neighbor_labels.append(cluster_cpu[0, j])
                # 下方邻居
                if j > 0 and j_bond_frozen_cpu[i, j-1]:
                    neighbor_indices.append((i, j-1))
                    neighbor_labels.append(cluster_cpu[i, j-1])
                # 上方邻居（周期性边界）
                elif j == array_len-1 and j_bond_frozen_cpu[i, j]:
                    neighbor_indices.append((i, 0))
                    neighbor_labels.append(cluster_cpu[i, 0])
                
                if not neighbor_indices:
                    # 创建新聚类
                    cluster_cpu[i, j] = current_label
                    prp_label_cpu[current_label] = current_label
                    current_label += 1
                else:
                    # 优化3: 使用numpy向量化操作找最小值
                    root_labels = np.array([self._proper_label(prp_label_cpu, label) 
                                           for label in neighbor_labels])
                    min_label = min(root_labels.min(), current_label)
                    
                    cluster_cpu[i, j] = min_label
                    
                    # 优化4: 批量更新唯一根标签，避免重复更新
                    unique_roots = np.unique(root_labels)
                    for root_label in unique_roots:
                        if root_label != min_label:
                            prp_label_cpu[root_label] = min_label
                    
                    # 优化5: 只更新需要更新的聚类位置
                    for (ni, nj), original_label in zip(neighbor_indices, neighbor_labels):
                        root_original = self._proper_label(prp_label_cpu, original_label)
                        if root_original != min_label:
                            cluster_cpu[ni, nj] = min_label
        
        # 将结果转回GPU（如果使用GPU）
        cluster = self._to_gpu(cluster_cpu)
        prp_label = self._to_gpu(prp_label_cpu)
        
        return cluster, prp_label