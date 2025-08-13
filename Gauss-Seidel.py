import numpy as np

def G_S(a, b, x, g, k):
    """
    Gauss-Seidel 迭代法求解线性方程组 Ax = b
    参数:
        a: 系数矩阵 (n×n)
        b: 右端向量 (n×1)
        x: 迭代初始值 (n×1)
        g: 精度阈值
        k: 最大迭代次数
    返回:
        x: 解向量
        times: 迭代次数
    """
    x = x.astype(float)
    m, n = a.shape
    times = 0
    
    # 检查输入
    if m != n:
        raise ValueError("系数矩阵必须为方阵")
    if m < n:
        raise ValueError("方程个数少于未知数，存在无穷多解")
    if np.any(np.diag(a) == 0):
        raise ValueError("矩阵对角元素不能为零")
    if len(b) != n or len(x) != n:
        raise ValueError("输入向量维度不匹配")
    
    while times < k:
        tempx = x.copy()
        for i in range(n):
            s1 = sum(a[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s1) / a[i][i]
        times += 1
        
        # 计算相对误差
        error = np.linalg.norm(b - np.dot(a, x)) / np.linalg.norm(b)
        if error < g:
            break
    
    if times >= k:
        print("警告：在最大迭代次数 {} 下未收敛".format(k))
    
    print("迭代次数：", times)
    print("解向量：", x)
    return x, times

# 测试代码
A=np.array([[20,4,6],[4,20,8],[6,8,20]])
b=np.array([10,-24,-22])
x=np.zeros(3)
x0=x.copy()
G_S(A,b,x,0.5*10**-4,200)