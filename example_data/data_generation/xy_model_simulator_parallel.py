"""
XY Model Simulator using Swendsen-Wang Algorithm with Parallel Computing Support

这是向后兼容的包装器，实际功能已拆分到以下模块：
- xy_utils.py: 工具类（内存池、三角函数表、并查集）
- xy_simulator_core.py: 核心模拟引擎
- xy_parallel.py: 并行处理和结果保存
- xy_simulator.py: 主入口类

为了兼容性，此文件保留了原有的导入接口
"""

# 从新模块导入主要类，保持向后兼容
from xy_simulator import ParallelXYModelSimulator

# 保持向后兼容的导出
__all__ = ['ParallelXYModelSimulator']
