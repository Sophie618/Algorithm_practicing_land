import pandas as pd
import numpy as np

dates=pd.date_range('20251015',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
print(df['A'])
print(df.A)#同上,打印A列
print(df[0:3])#打印前3行
print(df['20251015':'20251017'])#打印20251015到20251017行，同上

#----select by label:loc----
print(df.loc['20251015'])#打印20251015行
print(df.loc['20251015',['A','B']])#打印20251015行A列和B列

#----select by position:iloc----
print(df.iloc[0,1])#打印0行1列
print(df.loc[:,['A','B']])#打印所有行A列和B列
print(df.iloc[[1,3,5],1:3])#不连续打印1行3行5行1列和2列

print(df.iloc[:3,[0,2]])#打印前3行A列和C列 (使用位置索引)
# 或者使用 loc (如果你知道具体的标签)
# print(df.loc[dates[:3],['A','C']])#打印前3行A列和C列 (使用标签索引)

#----select by boolean----
print(df[df.A>4])#筛选A列大于4的行
print(df[df.A>4]['B'])#满足A列大于4的行中的B列
print(df[df.A>4][['B','C']])#满足A列大于4的行中的B列和C列

# #----operation----
# print(df.A+1)#打印A列加1
# print(df.A+df.B)#打印A列和B列相加
# print(df.A*df.B)#打印A列和B列相乘
# print(df.A/df.B)#打印A列和B列相除
# print(df.A%df.B)#打印A列和B列取模
# print(df.A**df.B)#打印A列和B列幂
# print(df.A//df.B)#打印A列和B列整除