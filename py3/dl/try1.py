import os
import sys
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train=torch.randn(100,3).to(device)
y_train=x_train*3+1+0.1*torch.randn(100,1).to(device)

class my_nn(nn.Module):
    def __init__(self,in_dim,hid_dim,out_dim):
        super().__init__()
        self.layer1=nn.Linear(in_dim,hid_dim).to(device)
        self.relu=nn.ReLU().to(device)
        self.layer2=nn.Linear(hid_dim,out_dim).to(device)

    def forward(self,x):
        x=self.layer1(x)
        x=self.relu(x)
        x=self.layer2(x)

        return x

def main(lr=0.005,epochs=20):
    losses=[]
    model=my_nn(3,5,1).to(device)
    loss_f=nn.MSELoss().to(device)
    optimizer=optim.SGD(model.parameters(),lr=0.005)
    
    for epoch in range(epochs):
        model.train()
        #前向传播
        outputs=model(x_train)
        loss=loss_f(outputs,y_train)

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if(epoch+1)%10==0:
            print(f'Epoch[{epoch+1}/{epochs}],loss:{loss.item():.4f}')

if __name__=='__main__':
    main()

