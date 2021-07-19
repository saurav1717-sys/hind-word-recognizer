#imports

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from segmentation import segment
from model_class import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#loading the model on cpu

model = torch.load("model.h5")                                                
model.to(device)

#ordinary function definition

def max(a,b):
    if a>=b:
        return a
    else:
        return b
    
def min(a,b):
    if a<=b:
        return a
    else:
        return b

#map to output the answer

relation = {0:1,10:6,11:8,12:12,13:13,14:16,15:20,1:25,2:27,3:28,4:31,5:32,6:33,7:35,8:37,9:41}  

#predict function

def predict(img):
    characters = segment(img)                
    answer = []
    for i in range(len(characters)):
        input = torch.tensor(characters[i]).view(1,1,100,100)
        input = input.type(torch.float32).to(device)
        output = torch.argmax(model(input),dim=1)
        temp = relation[int(output[0])]
        answer.append(temp)
    return answer






    
    

