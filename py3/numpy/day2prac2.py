import numpy as np

A=np.arange(14,2,-1).reshape(3,4)#arange左闭右开,(14,2,-1)表示从14到2，步长为-1，即倒序
print(A)
print(np.argmin(A))#求出A中最小值的索引
print(np.argmax(A))#求出A中最大值的索引
print(np.mean(A))#求出A中元素的平均值
print(A.mean())#与上一行效果相同
print(np.mean(A,axis=1))#求出A中每行元素的平均值
print(np.median(A))#求出A中元素的中位数
print(np.cumsum(A))#求出A中元素的累加和
print(np.diff(A))#求出A中元素的累差
print(np.nonzero(A))#求出A中非零元素的索引，返回一个元组，第一个列表元素为非零元素的行索引，第二个列表元素为列索引
print(np.sort(A))#求出A中元素的排序
print(np.transpose(A))#求出A的转置
print(A.T)#与上一行效果相同
print(np.clip(A.T,5,9))#将A.T中元素小于5的改为5，大于9的改为9，返回一个新的矩阵