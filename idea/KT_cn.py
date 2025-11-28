#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zhshang
"""
import matplotlib
matplotlib.use('Agg')  # 设置matplotlib使用非交互式后端，适用于服务器环境
import numpy as np
from numpy import linalg as LA  # 导入线性代数库
import matplotlib.pyplot as plt

# 模拟参数定义
L = 16  # 晶格尺寸：16x16的二维晶格
ESTEP = 1000  # 平衡步数：让系统达到热平衡的蒙特卡洛步数
STEP = 10000  # 测量步数：用于计算物理量的蒙特卡洛步数

J = 1  # 交换相互作用常数，J>0表示铁磁相互作用

# 初始化XY模型，随机生成自旋角度
def Init():
    return np.random.rand(L, L)*2*np.pi  # 返回一个LxL的矩阵，元素为[0, 2π)的随机角度
    #return np.ones([L, L])  # 均匀初始化的替代方案

# 周期性边界条件：返回下一个格点索引
def next(x):
    if x == L-1:
        return 0  # 如果当前是最后一个格点，下一个是第一个格点（周期性）
    else:
        return x+1

# 构建冻结键：根据Swendsen-Wang算法概率性地冻结相同自旋方向的键[1](@ref)
def FreezeBonds(Ising,T,S):
    iBondFrozen = np.zeros([L,L])  # 初始化水平方向的冻结键矩阵
    jBondFrozen = np.zeros([L,L])  # 初始化垂直方向的冻结键矩阵
    for i in np.arange(L):
        for j in np.arange(L):
            # 计算与右侧邻居形成冻结键的概率
            freezProb_nexti = 1 - np.exp(-2 * J * S[i][j] * S[next(i)][j] / T)
            # 计算与上方邻居形成冻结键的概率
            freezProb_nextj = 1 - np.exp(-2 * J * S[i][j] * S[i][next(j)] / T)
            # 如果自旋方向相同且随机数小于概率，则冻结水平键
            if (Ising[i][j] == Ising[next(i)][j]) and (np.random.rand() < freezProb_nexti):
                iBondFrozen[i][j] = 1
            # 如果自旋方向相同且随机数小于概率，则冻结垂直键
            if (Ising[i][j] == Ising[i][next(j)]) and (np.random.rand() < freezProb_nextj):
                jBondFrozen[i][j] = 1
    return iBondFrozen, jBondFrozen

# H-K算法（Hoshen-Kopelman）用于正确标记聚类标签[8](@ref)
def properlabel(prp_label,i):

    if not isinstance(i, (int, np.integer)):
        print(f"警告: i的类型是{type(i)}, 值={i}, 正在转换为整数")
        i = int(i)

    if i < 0 or i >= len(prp_label):
        print(f"错误: 索引{i}超出数组范围(0-{len(prp_label)-1})")
        return 0  # 返回一个安全的默认值
    
    i = int(i)  # 确保i是整数类型
    # 查找根标签：不断追溯直到找到代表该聚类的根标签
    while prp_label[i] != i:
        i = prp_label[i]
        if i < 0 or i >= len(prp_label):
            print(f"错误: 索引{i}在循环中超出范围")
            return 0
    return i

# Swendsen-Wang聚类查找：根据冻结键识别自旋聚类[1](@ref)
def clusterfind(iBondFrozen,jBondFrozen):
    cluster = np.zeros([L, L], dtype=int)  # 存储每个格点的聚类标签
    prp_label = np.zeros(L**2, dtype=int)  # 用于并查集数据结构的标签数组
    label = 0  # 当前可用的标签编号
    
    # 遍历所有格点
    for i in np.arange(L):
        for j in np.arange(L):
            bonds = 0  # 计算连接的冻结键数量
            ibonds = np.zeros(4,dtype=int)  # 存储连接的相邻格点的i坐标
            jbonds = np.zeros(4,dtype=int)  # 存储连接的相邻格点的j坐标

            # 检查与左侧格点(i-1,j)的冻结键
            if (i > 0) and iBondFrozen[i-1][j]:
                ibonds[bonds] = i-1
                jbonds[bonds] = j
                bonds += 1
            # 检查与右侧格点(i+1,j)的冻结键（考虑周期性边界）
            if (i == L-1) and iBondFrozen[i][j]:
                ibonds[bonds] = 0
                jbonds[bonds] = j
                bonds += 1
            # 检查与下方格点(i,j-1)的冻结键
            if (j > 0) and jBondFrozen[i][j-1]:
                ibonds[bonds] = i
                jbonds[bonds] = j-1
                bonds += 1
            # 检查与上方格点(i,j+1)的冻结键（考虑周期性边界）
            if (j == L-1) and jBondFrozen[i][j]:
                ibonds[bonds] = i
                jbonds[bonds] = 0
                bonds += 1

            # 如果没有连接的冻结键，创建新聚类
            if bonds == 0:
                cluster[i][j] = int(label)
                prp_label[int(label)] = int(label)  # 根标签指向自己
                label += 1
            # 有连接的冻结键，进行聚类合并
            else:
                minlabel = label  # 初始化最小标签
                # 查找相邻格点中的最小聚类标签
                for b in np.arange(bonds):
                    plabel = properlabel(prp_label,cluster[int(ibonds[b])][int(jbonds[b])])
                    if minlabel > plabel:
                        minlabel = plabel

                cluster[i][j] = minlabel
                # 链接所有相邻聚类的标签到最小标签
                for b in np.arange(bonds):
                    plabel_n = cluster[ibonds[b]][jbonds[b]]
                    prp_label[int(plabel_n)] = int(minlabel)  # 合并聚类
                    # 重新设置连接格点的标签
                    cluster[ibonds[b]][jbonds[b]] = minlabel
    return cluster, prp_label

# 翻转聚类自旋：以0.5概率翻转整个聚类[1](@ref)
def flipCluster(Ising,cluster,prp_label):
    for i in np.arange(L):
        for j in np.arange(L):
            # 重新标记所有聚类标签为正确的根标签
            cluster[i][j] = int(properlabel(prp_label, int(cluster[i][j])))
    sNewChosen = np.zeros(L**2)  # 标记聚类是否已选择新自旋方向
    sNew = np.zeros(L**2)  # 存储聚类的新自旋方向
    flips = 0  # 记录翻转的自旋数量，用于计算能量和磁化强度变化
    
    # 决定每个聚类的翻转情况
    for i in np.arange(L):
        for j in np.arange(L):
            label = cluster[i][j]
            randn = np.random.rand()
            # 如果该聚类尚未决定新自旋方向，随机决定（各50%概率）
            if (not sNewChosen[label]) and randn < 0.5:
                sNew[label] = +1
                sNewChosen[label] = True
            elif (not sNewChosen[label]) and randn >= 0.5:
                sNew[label] = -1
                sNewChosen[label] = True

            # 如果自旋方向改变，更新并计数
            if Ising[i][j] != sNew[label]:
                Ising[i][j] = sNew[label]
                flips += 1

    return Ising,flips

# Ising模型的Swendsen-Wang算法单步更新[1](@ref)
def oneMCstepIsing(Ising, S):
    [iBondFrozen, jBondFrozen] = FreezeBonds(Ising, T, S)  # 冻结键
    [SWcluster, prp_label] = clusterfind(iBondFrozen, jBondFrozen)  # 查找聚类
    [Ising, flips] = flipCluster(Ising, SWcluster, prp_label)  # 翻转聚类
    return Ising

# 将XY模型分解为两个Ising模型（沿投影方向proj）[1](@ref)
def decompose(XY,proj):
    x = np.cos(XY)  # 自旋的x分量
    y = np.sin(XY)  # 自旋的y分量
    # 旋转坐标系到投影方向
    x_rot = np.multiply(x,np.cos(proj))+np.multiply(y,np.sin(proj))
    y_rot = -np.multiply(x,np.sin(proj))+np.multiply(y,np.cos(proj))
    Isingx = np.sign(x_rot)  # x方向的Ising自旋（±1）
    Isingy = np.sign(y_rot)  # y方向的Ising自旋（±1）
    S_x = np.absolute(x_rot)  # x方向的幅度
    S_y = np.absolute(y_rot)  # y方向的幅度
    return Isingx, Isingy, S_x, S_y

# 将两个Ising模型组合回XY模型
def compose(Isingx_new,Isingy_new,proj,S_x, S_y):
    # 旋转回原始坐标系
    x_rot_new = np.multiply(Isingx_new,S_x)
    y_rot_new = np.multiply(Isingy_new,S_y)
    x_new = np.multiply(x_rot_new,np.cos(proj))-np.multiply(y_rot_new,np.sin(proj))
    y_new = np.multiply(x_rot_new,np.sin(proj))+np.multiply(y_rot_new,np.cos(proj))
    XY_new = np.arctan2(y_new,x_new)  # 计算新角度
    return XY_new

# XY模型的单步蒙特卡洛更新
def oneMCstepXY(XY):
    proj = np.random.rand()  # 随机选择投影方向
    [Isingx, Isingy, S_x, S_y] = decompose(XY, proj)  # 分解为两个Ising模型
    Isingx_new = oneMCstepIsing(Isingx, S_x)  # 更新x分量
    Isingy_new = oneMCstepIsing(Isingy, S_y)  # 更新y分量
    XY_new = compose(Isingx_new, Isingy_new, proj, S_x, S_y)  # 重新组合
    return XY_new

# 计算XY模型的能量和磁化强度
def EnMag(XY):
    energy = 0
    # 计算所有格点的总能量
    for i in np.arange(L):
        for j in np.arange(L):
            # 能量计算：-J∑S_i·S_j = -J∑cos(θ_i-θ_j)，考虑四个最近邻
            energy = energy - (np.cos(XY[i][j]-XY[(i-1)%L][j])+  # 左邻
                            np.cos(XY[i][j]-XY[(i+1)%L][j])+    # 右邻
                            np.cos(XY[i][j]-XY[i][(j-1)%L])+    # 下邻
                            np.cos(XY[i][j]-XY[i][(j+1)%L]))     # 上邻
    magx = np.sum(np.cos(XY))  # x方向总磁化强度
    magy = np.sum(np.sin(XY))  # y方向总磁化强度
    mag = np.array([magx,magy])
    return energy * 0.5, LA.norm(mag)/(L**2)  # 返回能量（每键计算了两次，故乘0.5）和标度化磁化强度

# Swendsen-Wang方法主函数：模拟XY模型在不同温度下的行为[1](@ref)
def SWang(T):
    XY = Init()  # 初始化系统
    
    # 热化过程：让系统达到平衡状态
    for step in np.arange(ESTEP):
        XY = oneMCstepXY(XY)
    
    # 测量过程：计算物理量的统计平均值
    E_sum = 0  # 能量累加器
    M_sum = 0  # 磁化强度累加器
    Esq_sum = 0  # 能量平方累加器
    Msq_sum = 0  ## 磁化强度平方累加器
    
    for step in np.arange(STEP):
        XY = oneMCstepXY(XY)  # 单步更新
        [E,M] = EnMag(XY)  # 计算能量和磁化强度

        E_sum += E
        M_sum += M
        Esq_sum += E**2
        Msq_sum += M**2

    # 计算平均值
    E_mean = E_sum/STEP/(L**2)  # 每格点平均能量
    M_mean = M_sum/STEP  # 平均磁化强度
    Esq_mean = Esq_sum/STEP/(L**4)  # 能量平方的平均值
    Msq_mean = Msq_sum/STEP  # 磁化强度平方的平均值

    return XY, E_mean, M_mean, Esq_mean, Msq_mean

# 主程序：在不同温度下模拟并计算物理量
M = np.array([])  # 存储各温度下的磁化强度
E = np.array([])  # 存储能量
M_sus = np.array([])  # 存储磁化率
SpcH = np.array([])  # 存储比热
Trange = np.linspace(0.1, 2.5, 10)  # 温度范围：0.1到2.5，10个点

# 遍历温度范围进行模拟
for T in Trange:
    [Ising, E_mean, M_mean, Esq_mean, Msq_mean] = SWang(T)
    M = np.append(M, np.abs(M_mean))  # 磁化强度取绝对值
    E = np.append(E, E_mean)
    M_sus = np.append(M_sus, 1/T*(Msq_mean-M_mean**2))  # 磁化率公式：χ = (⟨M²⟩-⟨M⟩²)/T
    SpcH = np.append(SpcH, 1/T**2*(Esq_mean-E_mean**2))  # 比热公式：Cv = (⟨E²⟩-⟨E⟩²)/T²

# 绘图部分[1](@ref)
T = Trange

# 绘制能量随温度变化图
plt.figure()
plt.plot(T, E, 'rx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle E \rangle$ per site $(J)$')
plt.savefig("E.pdf", format='pdf', bbox_inches='tight')

# 绘制比热随温度变化图
plt.figure()
plt.plot(T, SpcH, 'kx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$C_V$ per site $(\frac{J^2}{k_B^2})$')
plt.savefig("Cv.pdf", format='pdf', bbox_inches='tight')

# 绘制磁化强度随温度变化图
plt.figure()
plt.plot(T, M, 'bx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\langle|M|\rangle$ per site $(\mu)$')
plt.savefig("M.pdf", format='pdf', bbox_inches='tight')

# 绘制磁化率随温度变化图
plt.figure()
plt.plot(T, M_sus, 'gx-')
plt.xlabel(r'Temperature $(\frac{J}{k_B})$')
plt.ylabel(r'$\chi$ $(\frac{\mu}{k_B})$')
plt.savefig("chi.pdf", format='pdf', bbox_inches='tight')

plt.tight_layout()  # 自动调整子图参数
fig = plt.gcf()
plt.show()

# 保存数据到文件
np.savetxt('output.data',np.c_[T,E,SpcH,M,M_sus])

# 以下是测试单温度模拟的示例代码（已注释）
# ctitation:1
# T = 0.1
# [XY, E_mean, M_mean, Esq_mean, Msq_mean] = SWang(T)
# Cv = 1 / T**2 * (Esq_mean - E_mean**2)
# M_sus = 1 / T * (Msq_mean - M_mean**2)
# [E1,M1] = EnMag(XY)
# E2 = E1/(L**2)
# print(E_mean, E2, M_mean, M1, Cv, M_sus)
# # 绘制自旋构型图
# plt.figure()
# plt.matshow(XY,cmap='cool')
# plt.axis('off')