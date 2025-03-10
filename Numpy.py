import numpy as np

# 2. 将整数列表[1,2,3]转换为NumPy数组a，并查看数组对象a及a的类型
a = np.array([1, 2, 3])
print("数组a:", a)
print("数组a的类型:", type(a))

# 3. 将浮点数列表[1.2,2.3,3.4]转换为NumPy数组b，并查看b的类型
b = np.array([1.2, 2.3, 3.4])
print("数组b:", b)
print("数组b的类型:", type(b))

# 4. 生成2*3的全0数组
zeros_array = np.zeros((2, 3))
print("2*3的全0数组:\n", zeros_array)

# 5. 生成3*4的全1数组
ones_array = np.ones((3, 4))
print("3*4的全1数组:\n", ones_array)

# 6. 生成2*3未初始化的随机数数组
empty_array = np.empty((2, 3))
print("2*3未初始化的随机数数组:\n", empty_array)

# 7. 创建10-30之间，间隔步长为5，包含4个元素的一维数组
linspace_array = np.arange(10, 30, 5)
print("10-30之间，间隔步长为5的数组:", linspace_array)

# 8. 生成3*2的符合(0,1)均匀分布的随机数数组
uniform_array = np.random.uniform(0, 1, (3, 2))
print("3*2的符合(0,1)均匀分布的随机数数组:\n", uniform_array)

# 9. 生成0到2范围内长度为5的整数数组
int_array = np.arange(0, 3)
print("0到2范围内长度为5的整数数组:", int_array)

# 10. 生成长度为3，符合标准正态分布的随机数数组
normal_array = np.random.normal(0, 1, 3)
print("长度为3，符合标准正态分布的随机数数组:", normal_array)

# 11. 利用np.arange创建元素分别为[0,1,4,9,16,25,36,49,64,81]的一维数组a
a = np.array([i**2 for i in range(10)])
print("数组a:", a)

# 12. 获取数组a的第3个元素
element_3 = a[2]
print("数组a的第3个元素:", element_3)

# 13. 获取数组a的第2个到第4个数组元素
elements_2_to_4 = a[1:4]
print("数组a的第2个到第4个元素:", elements_2_to_4)

# 14. 翻转一维数组a
flipped_a = a[::-1]
print("翻转后的数组a:", flipped_a)

# 15. 创建一个3*3的符合[0,1]均匀分布的随机数数组b
b = np.random.uniform(0, 1, (3, 3))
print("3*3的随机数数组b:\n", b)

# 16. 获取b第2行第3列的数组元素
element_2_3 = b[1, 2]
print("b的第2行第3列元素:", element_2_3)

# 17. 获取b第2列数据
column_2 = b[:, 1]
print("b的第2列数据:", column_2)

# 18. 获取b第3列的前两行数据
column_3_first_two = b[:2, 2]
print("b的第3列的前两行数据:", column_3_first_two)

# 19. 创建一个3*4的符合[0,1]均匀分布的随机数组a，该元素乘以10后向下取整
random_array = np.random.uniform(0, 1, (3, 4))
floored_array = np.floor(random_array * 10)
print("3*4的随机数组乘以10后向下取整:\n", floored_array)

# 20. 将数组a展平
a = np.array([[1, 2, 3], [4, 5, 6]])
flattened_a = a.flatten()
print("展平后的数组a:", flattened_a)

# 21. 将数组a变换为2*6数组
reshaped_a = a.reshape(2, 3)
print("变换为 2*3 数组后的 a:\n", reshaped_a)
extended_a = np.tile(a, (1, 2))
print("扩展后的数组 a (2*6):\n", extended_a)

# 22. 求数组a的转置大小
transposed_a = a.T
print("数组a的转置:\n", transposed_a)

# 23. a.reshape(3,-1)中-1的含义是？
# -1 表示自动计算该维度的大小，使得总元素数不变。

# 24. 创建如下两个数组A和B
A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
print("数组A:\n", A)
print("数组B:\n", B)

# 25. 按行合并A数组和B数组
row_merged = np.vstack((A, B))
print("按行合并后的数组:\n", row_merged)

# 26. 按列合并A数组和B数组
column_merged = np.hstack((A, B))
print("按列合并后的数组:\n", column_merged)

# 27. 利用np.arange()创建如下数组C
C = np.arange(16).reshape(4, 4)
print("数组C:\n", C)

# 28. 按水平方向将数组C切分为两个数组
horizontal_split = np.hsplit(C, 2)
print("水平切分后的数组:")
for arr in horizontal_split:
    print(arr)

# 29. 按垂直方向将数组C切分为两个数组
vertical_split = np.vsplit(C, 2)
print("垂直切分后的数组:")
for arr in vertical_split:
    print(arr)