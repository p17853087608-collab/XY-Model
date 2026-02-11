"""
BKT相变温度分析
对每个温度点执行Bootstrap,计算均值和标准差
从Bootstrap均值曲线找50%概率点

这是向后兼容的包装器，实际功能已拆分到以下模块：
- bkt_config.py: 配置参数（数据路径、Bootstrap参数、输出目录、绘图样式）
- bkt_data_loader.py: 数据加载函数
- bkt_bootstrap.py: Bootstrap分析、BKT拟合、数据变换函数
- bkt_plotting.py: 绘图函数
- bkt_main.py: 主流程

为了兼容性，此文件保留了原有的main()函数
"""

import warnings
warnings.filterwarnings('ignore')

# 从新模块导入main函数
from bkt_main import main

# 向后兼容的导出
__all__ = ['main']


if __name__ == '__main__':
    main()
