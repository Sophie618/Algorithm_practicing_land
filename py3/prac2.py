import numpy as np
a=np.array([10,20,30,40])
b=np.arange(4)

print(a,b)
c=a+b
print(c)
d=b**2#平方
print(d)
e=10*np.sin(a)#cos、tan一样
print(e)

print(b)
print(b<3)#判断b内元素是否小于3，返回一个布尔矩阵