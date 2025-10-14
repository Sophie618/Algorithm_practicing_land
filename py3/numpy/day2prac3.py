import numpy as np
A=np.arange(3,15).reshape(3,4)
print(A)
print(A.flatten())#将A展平为1维数组
# print(A[2])#输出第三行
# print(A[2][1])#输出第三行第二列
# print(A[2,:])#输出第三行所有数
# print(A[:,1])#输出第二列所有数
# print(A[1,1:3])#输出第二行1-3列元素
# print(A[1,1:3].reshape(1,2))#将第二行第二列和第三列reshape为1行2列
# print(A[1:3,1:3])#输出第二行和第三行，第二列和第三列的元素

for row in A:
    print(row)

for column in A.flat:
    print(column)

for column in A.T:
    print(column)

for item in A:
    print(item)