    def _proper_label(self, prp_label: Any, i: int) -> int:
        """
        H-K算法（Hoshen-Kopelman）用于正确标记聚类标签
        
        参数:
            prp_label: 标签数组
            i: 标签索引
            
        返回:
            聚类的根标签
        """
        # 优化1: 缓存数组长度，避免重复调用len()
        if self.use_gpu:
            prp_label_cpu = cp.asnumpy(prp_label)
            i_cpu = int(i)
            array_len = len(prp_label_cpu)
            
            # 优化2: 提前边界检查，避免循环内重复检查
            if i_cpu < 0 or i_cpu >= array_len:
                return 0
            
            # 优化3: 路径压缩实现 - 查找过程中同时压缩路径
            root = i_cpu
            while prp_label_cpu[root] != root:
                root = prp_label_cpu[root]
            
            # 路径压缩：将路径上的所有节点直接指向根节点
            while prp_label_cpu[i_cpu] != i_cpu:
                next_i = prp_label_cpu[i_cpu]
                prp_label_cpu[i_cpu] = root
                i_cpu = next_i
            
            return root
        else:
            array_len = len(prp_label)
            
            if i < 0 or i >= array_len:
                return 0
            
            # 路径压缩实现
            root = i
            while prp_label[root] != root:
                root = prp_label[root]
            
            # 路径压缩
            while prp_label[i] != i:
                next_i = prp_label[i]
                prp_label[i] = root
                i = next_i
            
            return root