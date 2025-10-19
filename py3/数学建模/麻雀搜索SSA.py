import numpy as np
import random
import copy

#初始化麻雀种群
def initial(pop,dim,ub,lb):
    X=np.zeros([pop,dim])
    for i in range(pop):
        X[i,:]=np.random.uniform(low=lb[0],high=ub[0],size=(1,dim))
    return X,lb,ub
pop=50
dim=2#二维平面搜索
lb=np.min(-10)*np.ones([dim,1])
ub=np.max(10)*np.ones([dim,1])
X,lb,ub=initial(pop,dim,ub,lb)
# print(X.shape)

#定义适应度函数
def fun(X):
    O=0
    for i in X:
        O+=i**2
    return 0;

#计算适应度函数
def CaculateFitness(X,fun):
    pop=X.shape[0]
    fitness=np.zeros([pop,1])
    for i in range(pop):
        fitness[i]=fun(X[i,:])

