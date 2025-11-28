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
        
        print(f"开始XY模型模拟，共 {num_temperatures} 个温度点...")
        print(f"温度范围: {t_min:.2f} - {t_max:.2f}")
        print("=" * 50)
        
        # 遍历各温度点进行模拟
        for idx, temp in enumerate(temperature_array):
            temp_start = time.time()
            
            print(f"正在处理第 {idx+1}/{num_temperatures} 个温度点 (T = {temp:.3f})...", end=" ")
            
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
            
            # 记录时间并显示进度
            temp_time = time.time() - temp_start
            temperature_times[idx] = temp_time
            print(f"完成！耗时: {temp_time:.2f} 秒")
        
        print("=" * 50)
        
        # 总模拟时间
        total_time = time.time() - start_time
        avg_time_per_temp = temperature_times.mean()
        
        # 显示总耗时统计
        print(f"模拟完成！")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"平均每个温度点耗时: {avg_time_per_temp:.2f} 秒")
        print(f"最快温度点: {temperature_times.min():.2f} 秒")
        print(f"最慢温度点: {temperature_times.max():.2f} 秒")
        
        # 存储结果
        self.timing_data = {
            'total_time': total_time,
            'per_temperature_time': temperature_times,
            'avg_time_per_temp': avg_time_per_temp,
            'min_time': temperature_times.min(),
            'max_time': temperature_times.max()
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