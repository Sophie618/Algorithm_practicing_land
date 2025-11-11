from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
import numpy as np
import torch

class iris_dataloader(Dataset):
    def __init__(self,data_path):
        self.data_path=data_path#设置文件路径

        assert os.path.exists(self.data_path),"dataset does not exist."

        df=pd.read_csv(self.data_path,names=[0,1,2,3,4])

        d={"setosa":0,"versicolor":1,"virginica":2}#存储为字典形式
        df[4]=df[4].map(d)#把原数据内容按照字典内容定义进行替换

        data=df.iloc[:,:4]#第一个:代表从第0行取到最后一行，第二个:代表从第0列取到第4列
        label=df.iloc[:,4:]#标签就是最后一列

        data=(data-np.mean(data))/np.std(data) 
        #数据本身-其均值/其标准差->投射到均值为0，标准差为1的标准正态分布=》数据归一化操作<=>Z值化

        self.data=torch.from_numpy(np.array(data,dtype='float32'))
        self.label=torch.from_numpy(np.array(label,dtype='int32'))

        self.data_num=len(self.label)#统计数据数量
        print("当前数据集大小：",self.data_num)

    def __len__(self):
        return self.data_num

    def __getitem__(self,index):
        self.data=list(self.data)#用列表把数据作一个封装
        self.label=list(self.label)

        return self.data[index],self.label[index]#返回数据和标签


