import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import iris_dataloader

#初始化神经网络模型
class NN(nn.Module):
    def __init__(self,in_dim,hidden_dim1,hidden_dim2,out_dim):
        super().__init__()
        self.layer1=nn.Linear(in_dim,hidden_dim1)
        self.layer2=nn.Linear(hidden_dim1,hidden_dim2)
        self.layer3=nn.Linear(hidden_dim2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        return x

#定义计算环境
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#数据集的划分和加载。训练集，验证集和测试集
custom_dataset=iris_dataloader("py3/wx/yuanweihua/Iris_data.txt")#自定义数据集加载
train_size=int(len(custom_dataset)*0.7)
val_size=int(len(custom_dataset)*0.2)
test_size=len(custom_dataset)-train_size-val_size

train_dataset,val_dataset,test_dataset=torch.utils.data.random_split(custom_dataset,[train_size,val_size,test_size])

train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)
test_loader=DataLoader(test_dataset,batch_size=1,shuffle=False)
print("训练集的大小：",len(train_loader)*16,"验证集的大小：",len(val_loader),"测试集的大小：",len(test_loader))

#定义一个推理函数，来计算并返回准确率
def infer(model,dataset,device):
    model.eval()#设置为评估模式
    acc_num=0
    with torch.no_grad():#不计算梯度
        for data in dataset:
            datas,label=data
            outputs=model(datas.to(device))#将数据和标签移动到设备上
            predict_y=torch.max(outputs,dim=1)[1] #dim=1指在outputs的每一行中找最大值,torch.max()返回两个东西:result.values-最大值是多少,result.indices-最大值索引
            acc_num+=torch.eq(predict_y,label.to(device)).sum().item()
        acc=acc_num/len(dataset)
        return acc

#主函数,进行模型的训练和验证
def main(lr=0.005,epochs=20):#学习率和训练轮数的默认值
    model=NN(4,12,6,3).to(device)#四个参数其中输入和输出的数值是固定的
    loss_f=nn.CrossEntropyLoss()

    pg=[p for p in model.parameters() if p.requires_grad]
    #定义优化器
    optimizer=optim.Adam(pg,lr=lr)
    #权重文件存储路径
    save_path=os.path.join(os.getcwd(),"result/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

#开始训练
    for epoch in range(epochs):
        model.train()#设置为训练模式
        acc_num=torch.zeros(1).to(device)
        sample_num=0

        train_bar=tqdm(train_loader,file=sys.stdout,ncols=100)
        for datas in train_bar:
            data,label=datas
            label=label.squeeze(-1)
            sample_num+=data.shape[0]

            optimizer.zero_grad()#每次训练对优化器的梯度清零
            outputs=model(data.to(device))
            pred_class=torch.max(outputs,dim=1)[1]#torch.max返回值是一个元组，第一个元素是max的值第二个是max值的索引
            acc_num=torch.eq(pred_class,label.to(device)).sum().item()

            loss=loss_f(outputs,label.to(device).long())
            loss.backward()#反向传播
            optimizer.step()#更新参数

            train_acc=acc_num/sample_num
            train_bar.desc="train epoch[{}/{}],loss:{:.3f},acc:{:.3f}".format(epoch+1,epochs,loss,train_acc)

        val_acc=infer(model,val_loader,device)
        print("train epoch[{}/{}] loss{:.3f} train_acc{:.3f} val_acc{:.3f}".format(epoch+1,epochs,loss,train_acc,val_acc))
        torch.save(model.state_dict(),os.path.join(save_path,"nn.pth"))

        #每次数据集迭代之后要对初始化的指标清零
        train_acc=0.
        val_acc=0.
    print("Finished Training")

    test_acc=infer(model,test_loader,device)
    print("test_acc:",test_acc)

if __name__=="__main__":
    main()