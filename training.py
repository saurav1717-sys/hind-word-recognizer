import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(1)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((100,100)),
    transforms.Grayscale(1)
])

dataset = datasets.ImageFolder(r'C:\Users\saura\Data analysis\consonants',transform)
train = torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=True)

features = []
labels = []

for var in tqdm(train):
    labels.append(var[1])
    collect = []
    for i in range(len(var[0])):
        img = np.asarray(var[0][i]*255)
        ret, thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        thresh = thresh/255
        var[0][i] = torch.tensor(thresh).view(1,100,100)
    features.append(var[0])

train_x, test_x, train_y, test_y = train_test_split(features,labels,test_size=0.1)

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

net = Net().cuda()

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=0.001)

for epoch in range(30):
    for i in tqdm(range(len(train_x))):
        net.zero_grad()
        input = train_x[i].cuda()
        target = train_y[i].cuda()
        output = net(input)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
    print(f"Epoch : {epoch}, LOSS : {loss}")

net.eval()

total = 0
correct = 0
with torch.no_grad():
    for i in range(len(test_x)):
        input = test_x[i].cuda()
        target = test_y[i].cuda()
        output = torch.argmax(net(input),dim=1)
        total+=len(output)
        for j in range(len(output)):
            if output[j]==target[j]:
                correct+=1
                
print("Accuracy : ",100*correct/total)