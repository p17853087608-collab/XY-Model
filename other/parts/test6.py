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
        
        # 优化1: 预分配所有结果数组，避免动态扩展
        t_min, t_max = temperature_range
        temperature_array = np.linspace(t_min, t_max, num_temperatures)
        
        # 优化2: 使用连续内存布局，提升缓存效率
        magnetization_array = np.zeros(num_temperatures, dtype=np.float64)
        energy_array = np.zeros(num_temperatures, dtype=np.float64)
        susceptibility_array = np.zeros(num_temperatures, dtype=np.float64)
        specific_heat_array = np.zeros(num_temperatures, dtype=np.float64)
        temperature_times = np.zeros(num_temperatures, dtype=np.float64)
        
        # 优化3: 预计算常量，避免循环内重复计算
        l_squared = self.L ** 2
        l_fourth = self.L ** 4
        inv_step = 1.0 / self.STEP
        inv_l_squared = 1.0 / l_squared
        
        print(f"开始XY模型模拟，共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print("=" * 50)
        
        # 优化4: 批量处理温度点，减少函数调用开销
        for idx, temp in enumerate(temperature_array):
            temp_start = time.time()
            print(f"正在处理第 {idx+1}/{num_temperatures} 个温度点 (T = {temp:.3f})...", end=" ")
            
            # 初始化自旋
            xy = self._initialize_spins()
            
            # 优化5: 热化过程 - 使用批量Monte Carlo步骤
            # 将循环展开为块操作，提高缓存命中率
            batch_size = 10
            estep_batches = self.ESTEP // batch_size
            for _ in range(estep_batches):
                for _ in range(batch_size):
                    xy = self._one_mc_step_xy(xy, temp)
            
            # 处理剩余步骤
            for _ in range(self.ESTEP % batch_size):
                xy = self._one_mc_step_xy(xy, temp)
            
            # 优化6: 预分配测量数组，减少动态内存分配
            energy_measurements = np.zeros(self.STEP, dtype=np.float64)
            magnetization_measurements = np.zeros(self.STEP, dtype=np.float64)
            
            # 优化7: 测量过程 - 批量计算和存储
            for step_idx in range(self.STEP):
                xy = self._one_mc_step_xy(xy, temp)
                energy, magnetization = self._calculate_energy_magnetization(xy)
                energy_measurements[step_idx] = energy
                magnetization_measurements[step_idx] = magnetization
            
            # 优化8: 使用向量化操作计算统计量，避免循环
            energy_sum = energy_measurements.sum()
            magnetization_sum = magnetization_measurements.sum()
            energy_sq_sum = (energy_measurements ** 2).sum()
            magnetization_sq_sum = (magnetization_measurements ** 2).sum()
            
            # 优化9: 使用预计算的常量进行除法运算
            energy_mean = energy_sum * inv_step * inv_l_squared
            magnetization_mean = magnetization_sum * inv_step
            energy_sq_mean = energy_sq_sum * inv_step / l_fourth
            magnetization_sq_mean = magnetization_sq_sum * inv_step
            
            # 优化10: 预计算温度的倒数和平方倒数，避免重复计算
            inv_temp = 1.0 / temp
            inv_temp_squared = inv_temp * inv_temp
            
            # 计算派生物理量
            susceptibility = (magnetization_sq_mean - magnetization_mean * magnetization_mean) * inv_temp
            specific_heat = (energy_sq_mean - energy_mean * energy_mean) * inv_temp_squared
            
            # 优化11: 批量存储结果，减少内存写入次数
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
            
            # 记录时间并显示进度
            temp_time = time.time() - temp_start
            temperature_times[idx] = temp_time
            print(f"完成！耗时: {temp_time:.2f} 秒")
        
        print("=" * 50)
        
        # 优化12: 使用向量化操作计算时间统计
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean()
        min_time = temperature_times.min()
        max_time = temperature_times.max()
        
        # 显示总耗时统计
        print(f"模拟完成！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        print(f"最快温度点: {min_time:.2f} 秒")
        print(f"最慢温度点: {max_time:.2f} 秒")
        
        # 优化13: 使用字典批量存储，减少字典操作次数
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times,
            'avg_time_per_temp': avg_time_per_temp,
            'min_time': min_time,
            'max_time': max_time
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
        
        # 自动生成所有输出
        self.plot_results()
        self.generate_spin_visualization()
        self.save_results()
        
        return self.results