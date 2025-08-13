import numpy as np

def is_zero_vector(v):
    """
    检查输入是否为零向量
    参数:
        v: 输入向量（NumPy 数组）
    返回:
        bool: True 如果是零向量，False 否则
    异常:
        TypeError: 如果输入不是一维向量
    """
    if v.ndim != 1 and not np.isscalar(v):
        raise TypeError("输入必须为一维向量")
    return not np.any(v)

def power_iteration(A, x, k=200, tol=1e-6):
    """
    使用幂迭代法计算矩阵的主特征值和特征向量
    参数:
        A: 方阵（n×n NumPy 数组）
        x: 初始向量（n×1 NumPy 数组）
        k: 最大迭代次数（默认 200）
        tol: 收敛精度（默认 1e-6）
    返回:
        eigval: 主特征值
        eigvec: 对应的归一化特征向量
    异常:
        ValueError: 如果 A 不是方阵、x 是零向量或维度不匹配
    """
    # 输入验证
    if not isinstance(A, np.ndarray) or A.shape[0] != A.shape[1]:
        raise ValueError("矩阵 A 必须为方阵")
    if is_zero_vector(x):
        raise ValueError("初始向量 x 不能为零向量")
    if A.shape[1] != x.shape[0]:
        raise ValueError("矩阵 A 的列数必须等于向量 x 的长度")
    
    x = x.astype(float)
    for i in range(k):
        # 矩阵-向量乘法
        x_new = np.matmul(A, x)
        # 使用 2-范数归一化
        norm = np.linalg.norm(x_new)
        if norm == 0:
            raise ValueError("迭代向量变为零向量")
        eigvec = x_new / norm
        # 检查收敛性
        if np.linalg.norm(x - eigvec) < tol:
            break
        x = eigvec
    
    # 估算特征值（瑞利商）
    eigval = np.dot(eigvec, np.matmul(A, eigvec)) / np.dot(eigvec, eigvec)
    
    return eigval, eigvec

# 测试代码
if __name__ == "__main__":
    A = np.array([[2, 1], [1, 2]], dtype=float)
    x = np.array([1, 0], dtype=float)
    eigval, eigvec = power_iteration(A, x)
    print("主特征值：", eigval)
    print("特征向量：", eigvec)