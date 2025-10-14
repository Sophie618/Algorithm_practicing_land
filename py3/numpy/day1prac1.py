import numpy as np

# 矩阵基本属性
array=np.array([[1,2,3],[2,3,4]]) #把一个列表转换为矩阵
print(array)
print('number of dim:',array.ndim) # 输出矩阵的维度
print('shape:',array.shape) # 输出矩阵的形状（行列数）
print('size:',array.size,'\n') # 输出矩阵的元素个数

# 矩阵的创建
a=np.array([1,2,3],dtype=np.float32) # 创建一维矩阵，元素类型为float32
print(a.dtype) # 输出矩阵的元素类型（int32、int64、float32、float64）

a=np.array([[2,23,4],[2,32,4]]) # 创建二维矩阵
print(a)

a=np.zeros((3,4)) # 创建一个3行4列默认值为0的矩阵
b=np.ones((3,4)) # 默认值为1的矩阵
c=np.empty((3,4)) # 默认值几乎接近0的矩阵
d=np.arange(10,20,2) #10到20的有序矩阵，步长为2
e=np.arange(12).reshape((3,4)) # 创建一个3行4列的矩阵，元素为0到11,并转换为3行4列的矩阵
f=np.linspace(1,10,5) # 创建一个1到10的矩阵，元素个数为5,自动根据元素个数计算步长
print(a,b,c,d,e,f)