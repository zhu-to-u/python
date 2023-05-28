import numpy as np

# 输入判断矩阵A
A = np.array([[1, 1, 4, 1/3, 3],
              [1, 1, 4, 1/3, 3],
              [1/4, 1/4, 1, 1/3, 1/2],
              [3, 3, 3, 1, 3],
              [1/3, 1/3, 2, 1/3, 1]])

# 方法1：算术平均法求权重
# 第一步：将判断矩阵按列归一化
sum_A = np.sum(A, axis=0)
stand_A = A / sum_A

# 第二步：将归一化的各列相加（按行求和）
sum_rows = np.sum(stand_A, axis=1)

# 第三步：将相加后的向量中每个元素除以n得到权重向量
arithmetic_weights = sum_rows / len(A)

print("算术平均法求权重的结果为：")
print(arithmetic_weights)

# 方法2：几何平均法求权重
# 第一步：将A的元素按行相乘得到一个新的列向量
product_A = np.prod(A, axis=1)

# 第二步：将新的向量的每个分量开n次方
product_n_A = product_A ** (1 / len(A))

# 第三步：对该列向量进行归一化即可得到权重向量
geometric_weights = product_n_A / np.sum(product_n_A)

print("几何平均法求权重的结果为：")
print(geometric_weights)

# 方法3：特征值法求权重
# 第一步：求出矩阵A的最大特征值以及其对应的特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
max_eigenvalue = np.max(eigenvalues)
max_eigenvalue_index = np.argmax(eigenvalues)
max_eigenvector = eigenvectors[:, max_eigenvalue_index]

# 第二步：对求出的特征向量进行归一化即可得到权重
eigenvector_weights = max_eigenvector / np.sum(max_eigenvector)

print("特征值法求权重的结果为：")
print(eigenvector_weights)

# 计算一致性比例CR
n = len(A)
CI = (max_eigenvalue - n) / (n - 1)
RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
CR = CI / RI[n]

print("一致性指标CI =", CI)
print("一致性比例CR =", CR)

if CR < 0.10:
    print("因为CR < 0.10，所以该判断矩阵A的一致性可以接受！")
else:
    print("注意：CR >= 0.10，因此该判断矩阵A需要进行修改！")
