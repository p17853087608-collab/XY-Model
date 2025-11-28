import cupy as cp
import numpy as np

# 1. 创建数组并验证计算
x_gpu = cp.arange(10)  # 在GPU上创建数组
print("CuPy array:", x_gpu)

x_cpu = np.arange(10)  # 在CPU上创建等价的NumPy数组
l2_gpu = cp.linalg.norm(x_gpu)  # 使用CuPy计算范数
l2_cpu = np.linalg.norm(x_cpu)  # 使用NumPy计算范数

print("GPU L2 norm:", l2_gpu)
print("CPU L2 norm:", l2_cpu)

# 2. 数据迁移
# 将数据从CPU（主机）复制到GPU（设备）
a_cpu = np.array([1, 2, 3])
a_gpu = cp.asarray(a_cpu)

# 将数据从GPU复制回CPU
result_cpu = cp.asnumpy(a_gpu)
# 或者使用 .get() 方法
result_cpu_alt = a_gpu.get()

print(cp.__version__)  # 打印CuPy的版本号，确认具体版本