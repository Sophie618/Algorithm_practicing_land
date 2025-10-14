import numpy as np
A=np.arange(12).reshape(3,4)
B=A#赋值,A和B是同一个对象,互相关联
# A[0]=11
# print(B)#B也会改变
# C=A.view()#浅拷贝,shallow copy,A和C互相关联------可以直接用赋值实现，不用管
# # print(C is A)#判断C和A是否互相关联
# # A[0]=12
# # print(A)
# # print(C)
# # C[0]=13
# # print(A)
# # print(C)
D=A.copy()#深拷贝,deep copy,A和D不互相关联
# print(D is A)#判断D和A是否互相关联
# D[0]=13
# print(A)
# print(D)
# A[0]=14
# print(A)
# print(D)
#----区分拷贝和赋值----

import numpy as np

A = np.arange(12).reshape(3,4)
print("初始 A:", A)

D = A.copy()
print("D is A:", D is A)  # 应该是 False

D[0] = 13
print("修改D后 A:", A)    # A 应该不变
print("修改D后 D:", D)    # D 应该变了

A[0] = 14
print("修改A后 A:", A)    # A 变了
print("修改A后 D:", D)    # D 不变