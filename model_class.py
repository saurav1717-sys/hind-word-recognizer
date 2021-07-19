import torch
import torch.nn as nn
import torch.nn.functional as F

#class definition of the model

class Net(nn.Module):                                                                             
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,256,3,padding=1)
        self.conv5 = nn.Conv2d(256,512,3,padding=1)
        self.conv6 = nn.Conv2d(512,1024,3,padding=1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        self.batch_norm5 = nn.BatchNorm2d(512)
        self.batch_norm6 = nn.BatchNorm2d(1024)
        self.to_linear = None
        x = torch.randn(1,1,100,100)
        self.convs(x)
        self.fc1 = nn.Linear(self.to_linear,16)
    
    def convs(self,x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.max_pool2d(x,(2,2))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = F.max_pool2d(x,(2,2))
        if self.to_linear == None:
            self.to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self,x):
        x = self.convs(x)
        x = x.view(-1,self.to_linear)
        x = F.log_softmax(self.fc1(x),dim=1)
        return x