# 基于Swendsen-Wang聚类算法的XYModel的蒙特卡洛模拟

#### 介绍
XY模型模拟器是一个使用Swendsen-Wang聚类算法模拟二维XY模型的Python工具包。本模拟器支持CPU和GPU加速计算，能够高效计算不同温度下的物理量，如能量、磁化强度、比热和磁化率等。

#### 改动
1.显示代码运行进度,温度和耗时  
2.自旋可视化图由彩色图表示   
3.并行批量生成自旋图功能（多线程+GPU加速）   

#### 使用指南  
xy_model_simulator 文件夹是 xy模型模拟器。 包括模型，并行批量脚本，测试脚本，使用文档  
  
idea文件夹是 我的灵感来源

environment.yml 是我的工作环境配置  
在另一台机器上，使用该YAML文件可以重新创建出完全相同的环境。  
核心命令 (恢复):	
```
conda env create -f environment.yml
```

simulation_results_xxxx文件夹 保存了我的测试结果   



