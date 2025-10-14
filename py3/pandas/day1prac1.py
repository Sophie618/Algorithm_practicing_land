import pandas as pd
import numpy as np

s=pd.Series([1,2,np.nan,44,5])#np.nan=null
print(s)
print(s.isnull())
print(s.notnull())
print(s.fillna(0))#填充缺失值
print(s.dropna())#删除缺失值
print(s.fillna(method='ffill'))#向前填充
print(s.fillna(method='bfill'))#向后填充
print(s.fillna(method='ffill',limit=1))#向前填充，限制填充次数
print(s.fillna(method='bfill',limit=1))#向后填充，限制填充次数

date_range=pd.date_range('2025-01-01',periods=10)
print(date_range)
df=pd.DataFrame(np.random.randn(10,4),index=date_range,columns=list('ABCD'))
print(df)
df1=pd.DataFrame(np.arange(12).reshape(3,4))
print(df1)
df2=pd.DataFrame({'A':1.,
            'B':pd.Timestamp('20250102'),
            'C':pd.Series(1,index=list(range(4)),dtype='float32'),
            'D':np.array([3]*4,dtype='int32'),
            'E':pd.Categorical(['test','train','test','train']),
            'F':'foo'
            })
print(df2)
print(df2.dtypes)
print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe())#描述性统计
print(df2.T)#转置
print(df2.sort_index(axis=1,ascending=False))#按列索引排序,降序
print(df2.sort_index(axis=0,ascending=False))#按行索引排序
print(df2.sort_index(axis=1,ascending=True))#按列索引排序,升序
print(df2.sort_index(axis=0,ascending=True))#按行索引排序

df2.sort_values(by='A')#按A列排序
df2.sort_values(by='A',ascending=False)#按A列排序，降序
df2.sort_values(by='A',ascending=True)#按A列排序，升序
df2.sort_values(by='A',ascending=True,inplace=True)#按A列排序，升序，修改原数据
