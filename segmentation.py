import cv2
from torchvision import transforms
import torch
import numpy as np

#function to separate each character from the image

def segment(img):                                                                    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
    h = thresh.shape[0]
    w = thresh.shape[1]
    x1 = h
    x2 = 0
    y1 = w
    y2 = 0
    for i in range(h):
        for j in range(w):
            if thresh[i,j]==255:
                x1 = min(x1,i)
                x2 = max(x2,i)
                y1 = min(y1,j)
                y2 = max(y2,j)
                
    crop_img = thresh[max(0,x1-10):min(h,x2+10),max(0,y1-10):min(w,y2+10)]
    output = crop_img.copy()
    h = output.shape[0]
    w = output.shape[1]
    for i in range(w):
        for j in range(h):
            if crop_img[j,i]==255:
                for k in range(int(h/20)):
                    output[min(h-1,j+k),i]=0
                break
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    real = []
    for i in range(len(contours)):
        if hierarchy[0][i][3]==-1:
            real.append(i)
    mx = 0
    for var in real:
        pm = cv2.arcLength(contours[var],False)
        mx = max(mx,pm)
    final = []
    for var in real:
        pm = cv2.arcLength(contours[var],False)
        if pm>mx/5:
            final.append(var)
    images = []
    for var in final:
        temp = contours[var]
        x1 = w
        x2 = 0
        y1 = h
        y2 = 0
        for items in temp:
            x1 = min(x1,items[0][0])
            x2 = max(x2,items[0][0])
            y1 = min(y1,items[0][1])
            y2 = max(y2,items[0][1])
        
        crop = crop_img[max(0,y1-int(h/15)):min(h,y2+int(h/15)),max(0,x1-int(w/50)):min(w,x2+int(w/50))]
        images.append((x1,crop))  
    final_images = []      
    transform = transforms.Resize((100,100))
    for var in images:
        output = transform(torch.tensor(var[1]).view(1,var[1].shape[0],var[1].shape[1]))
        new_img = np.asarray(output[0])
        ret2, thresh2 = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY)
        final_images.append((var[0],thresh2/255))
    final_images.sort()
    to_return = []
    for var in final_images:
        to_return.append(var[1])  
    return to_return 