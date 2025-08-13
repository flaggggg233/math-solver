# 数值计算项目集
## 项目概述
本仓库包含数值分析课程项目，展示线性代数和数值计算能力。
- **Gauss-Seidel 求解器**：求解线性方程组，误差控制至 \(10^{-6}\)。
- **幂迭代法**：计算矩阵主特征值和特征向量，收敛精度 \(10^{-6}\)。
- **技术栈**：Python, NumPy

## 使用方法
```python
import numpy as np
from gauss_seidel import G_S
from power_iteration import power_iteration

# 示例：Gauss-Seidel
A=np.array([[20,4,6],[4,20,8],[6,8,20]])
b=np.array([10,-24,-22])
x=np.zeros(3)
x0=x.copy()
G_S(A,b,x,0.5*10**-4,200)
print("Gauss-Seidel 解向量：", eigvec)

# 示例：幂迭代法
A = np.array([[2, 1], [1, 2]])
x = np.array([1, 0])
eigval, eigvec = power_iteration(A, x, 100, 1e-6)
print("主特征值：", eigval, "特征向量：", eigvec)