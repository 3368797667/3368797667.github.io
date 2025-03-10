import numpy as np

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# 欧氏距离
euclidean_distance = np.linalg.norm(x - y)
print("欧氏距离:", euclidean_distance)

# 曼哈顿距离
manhattan_distance = np.sum(np.abs(x - y))
print("曼哈顿距离:", manhattan_distance)

# 闵可夫斯基距离 (p=3 作为示例)
p = 3
minkowski_distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
print("闵可夫斯基距离 (p=3):", minkowski_distance)

# 余弦距离
cosine_similarity = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
cosine_distance = 1 - cosine_similarity
print("余弦距离:", cosine_distance)

import numpy as np

# 1. 生成数组元素是0至9的一维数组MyArray
MyArray = np.arange(10)
print("1. MyArray:", MyArray)

# 2. 对MyArray数组元素分别计算均值、标准差、总和、最大值
mean_value = np.mean(MyArray)
std_value = np.std(MyArray)
sum_value = np.sum(MyArray)
max_value = np.max(MyArray)
print("2. 均值:", mean_value, "标准差:", std_value, "总和:", sum_value, "最大值:", max_value)

# 3. 利用数组方法.cumsum()计算MyArray数组元素的当前累计和
cumsum_array = MyArray.cumsum()
print("3. 累计和:", cumsum_array)

# 4. 利用NumPy函数sort()对数组元素开平方
sqrt_array = np.sqrt(MyArray)
print("4. 开平方后的数组:", sqrt_array)

# 5. 利用NumPy函数seed()指定随机数种子
np.random.seed(42)

# 6. 利用NumPy函数random.randn()生成包含10个元素且服从标准正态分布的一维数组
random_array = np.random.randn(10)
print("6. 标准正态分布数组:", random_array)

# 7. 利用NumPy函数sort()对数组元素排序，排序结果不覆盖原数组内容
sorted_array = np.sort(random_array)
print("7. 排序后的数组:", sorted_array)
print("   原数组:", random_array)

# 8. 利用NumPy函数where()依次对数组元素进行逻辑判断
condition_array = np.where(MyArray > 5, MyArray, -1)
print("8. 逻辑判断后的数组:", condition_array)

# 9. 利用NumPy的random.normal()函数生成2行5列的二维数组，数组元素服从均值为5、标准差为1的正态分布
normal_array = np.random.normal(5, 1, (2, 5))
print("9. 正态分布二维数组:\n", normal_array)

# 10. 利用eye()函数生成一个5行5列的单位矩阵Y
Y = np.eye(5)
print("10. 单位矩阵 Y:\n", Y)

# 11. 利用dot()函数计算矩阵X和矩阵Y的矩阵乘积
X = np.random.rand(5, 5)  # 生成一个5x5的随机矩阵X
dot_product = np.dot(X, Y)
print("11. 矩阵乘积 X * Y:\n", dot_product)

from numpy.linalg import inv, svd, eig

# 示例矩阵
A = np.array([[4, 7], [2, 6]])

# 计算矩阵的逆
A_inv = inv(A)
print("矩阵的逆:\n", A_inv)

# 对矩阵进行奇异值分解
U, S, Vh = svd(A)
print("奇异值分解的 U:\n", U)
print("奇异值分解的 S:\n", S)
print("奇异值分解的 Vh:\n", Vh)

# 计算矩阵的特征值和特征向量
eigenvalues, eigenvectors = eig(A)
print("特征值:\n", eigenvalues)
print("特征向量:\n", eigenvectors)

# 生成一个 3x3 的二维数组 X，元素服从标准正态分布
X = np.random.randn(3, 3)
print("二维数组 X:\n", X)

# 1) 生成 mat，mat 为 X 的转置矩阵与 X 的矩阵乘积
mat = np.dot(X.T, X)
print("mat:\n", mat)

# 2) 计算 mat 矩阵的逆
mat_inv = inv(mat)
print("mat 的逆:\n", mat_inv)

# 3) 计算矩阵 mat 的特征值和特征向量
eigenvalues, eigenvectors = eig(mat)
print("mat 的特征值:\n", eigenvalues)
print("mat 的特征向量:\n", eigenvectors)

# 4) 对矩阵 mat 做奇异值分解
U, S, Vh = svd(mat)
print("mat 的奇异值分解的 U:\n", U)
print("mat 的奇异值分解的 S:\n", S)
print("mat 的奇异值分解的 Vh:\n", Vh)


# 定义向量
vectors = np.array([[3, 4], [5, 6], [2, 2], [8, 4]])

# 计算协方差矩阵
cov_matrix = np.cov(vectors, rowvar=False)

# 计算协方差矩阵的逆
cov_matrix_inv = np.linalg.inv(cov_matrix)

# 计算均值向量
mean_vector = np.mean(vectors, axis=0)

# 定义马氏距离函数
def mahalanobis_distance(x, y, cov_inv):
    diff = x - y
    return np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

# 计算所有向量对之间的马氏距离
distances = []
for i in range(len(vectors)):
    for j in range(i + 1, len(vectors)):
        dist = mahalanobis_distance(vectors[i], vectors[j], cov_matrix_inv)
        distances.append((i, j, dist))

# 输出结果
print("马氏距离:")
for i, j, dist in distances:
    print(f"向量 {i} 和向量 {j} 之间的马氏距离: {dist}")