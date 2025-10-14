import numpy as np
A=np.arange(12).reshape(3,4)
print(A)

#----等量分割----
print(np.split(A,2,axis=1))#将A按列分成2份
print(np.split(A,3,axis=0))#将A按行分成3份
#----不等量分割----
print(np.array_split(A,3,axis=1))#将A按列分成3份
print(np.array_split(A,2,axis=0))#将A按行分成2份
#----垂直分割----
print(np.vsplit(A,3))#将A按行分成3份vertical split
#----水平分割----
print(np.hsplit(A,2))#将A按列分成2份horizontal split