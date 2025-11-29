from xy_plotter import XYPlotter

# 创建绘图器实例
plotter = XYPlotter()

# 加载结果文件
if plotter.load_results("simulation_results.txt"):
    # 绘制所有物理量图像
    plotter.plot_all(file_format='pdf')