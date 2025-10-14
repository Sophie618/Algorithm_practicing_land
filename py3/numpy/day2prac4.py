import numpy as np
A=np.array([1,1,1])
B=np.array([2,2,2])

C=np.vstack((A,B))#将A和B垂直合并 vertical stack
D=np.hstack((A,A,B))#将2个A和B水平合并 horizontal stack
print(C)
print(D)
print(A.shape,B.shape,C.shape,D.shape)
print(A[np.newaxis,:])#将A变为1行3列
print(B[np.newaxis,:])
print(np.concatenate((A,B,A),axis=0))#将A、B、A垂直合并

A=np.array([1,1,1])[:,np.newaxis]
B=np.array([2,2,2])[:,np.newaxis]
print(A)
print(B)
print(np.concatenate((A,B,A,B),axis=1))#水平合并


