import numpy as np
a=np.random.random((2,4))

print(a)
print(np.sum(a))
print(np.sum(a,axis=0))#axis=0表示按列求和
print(np.max(a,axis=1))#axis=1表示求出每行中的最大值
print(np.min(a))