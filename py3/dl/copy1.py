import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN,self).__init__()
        self.fc1=nn.Linear(3,5)
        self.fc2=nn.Linear(5,1)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

#创建模型
model = SimpleNN()

#随机输入
input_data=torch.randn(1,3)
output=model(input_data)
print(output)