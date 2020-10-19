'''
BagData.py
'''
import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2
#from onehot import onehot

#transform是对图像进行预处理、数据增强等。Compose将多个处理步骤整合到一起。
#ToTensor：将原始取值0-255像素值，归一化为0-1
#Normalize：用像素值的均值和标准偏差对像素值进行标准化
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class BagDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('./bags/last'))

    def __getitem__(self, idx):
        #读取原图
        img_name = os.listdir('./bags/last')[idx]
        imgA = cv2.imread('./bags/last/'+img_name)
        imgA = cv2.resize(imgA, (160, 160))
        
        #读取标签图，即二值图
        imgB = cv2.imread('/bags/bags/last_msk/'+img_name, 0)
        imgB = cv2.resize(imgB, (160, 160))

        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2) #因为此代码是二分类问题，即分割出手提包和背景两样就行，因此这里参数是2
        imgB = imgB.transpose(2,0,1) #imgB不经过transform处理，所以要手动把(H,W,C)转成(C,H,W)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA) #一转成向量后，imgA通道就变成(C,H,W)
        return imgA, imgB

bag = BagDataset(transform)
train_size = int(0.9 * len(bag))    #整个训练集中，90%为训练集
test_size = len(bag) - train_size

train_dataset, test_dataset = random_split(bag, [train_size, test_size]) #按照上述比例(9:1)划分训练集和测试集
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)

if __name__ =='__main__':
    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
