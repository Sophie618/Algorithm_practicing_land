import pandas as pd
import numpy as np

dates=pd.date_range('20251015',periods=6)
df=pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
df.iloc[0,1]=np.nan
df.iloc[1,2]=np.nan
raw_train_data=df
raw_test_data=df

# #----na处理 how={'any','all'}
# print(df.dropna(axis=0,how='any'))#删除有NaN的行
# print(df.dropna(axis=1,how='all'))#删除全是NaN的列
# print(df.fillna(value=0))#填充NaN为0
# print(df.isnull())#判断是否为NaN
# print(df.notnull())#判断是否不为NaN
# print(np.any(df.isnull())==True)#判断是否有NaN
# print(np.any(df.notnull())==True)#判断是否有不为NaN

#----thresh=n
print(df.dropna(axis=0,thresh=4))#删除有效值少于4个的行
print(df.dropna(axis=1,thresh=3))#删除有效值少于3个的列

#dropna 参数详解：
df.dropna(
    axis=0,           # 0=删除行，1=删除列
    how='any',        # 'any'=有任何NaN就删除，'all'=全部NaN才删除
    thresh=3,         # 有效值少于3个就删除
    subset=['A','B'], # 只检查指定列
    inplace=False     # True=直接修改，False=返回副本
)

# 常见的数据清洗流程
def clean_data(df):
    # 1. 删除缺失值过多的行（超过50%）
    df = df.dropna(axis=0, thresh=len(df.columns)//2)
    
    # 2. 删除缺失值过多的列（超过80%）
    df = df.dropna(axis=1, thresh=len(df)*0.2)
    
    # 3. 填充剩余的缺失值
    df = df.fillna(df.mean())  # 用均值填充数值列
    df = df.fillna(df.mode().iloc[0])  # 用众数填充分类列
    
    return df

# 在训练前清洗数据
train_data = clean_data(raw_train_data)
test_data = clean_data(raw_test_data)