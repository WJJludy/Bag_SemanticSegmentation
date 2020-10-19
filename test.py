'''
t.py
'''
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import cv2
from dataset import test_dataloader, train_dataloader
from model import FCNs,VGGNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('./results/fcn_model_95.pt')  # 加载模型
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


if __name__ =='__main__':
    img_name =r'c:/users/w1998/desktop/a.png'  #预测的图片
    imgA = cv2.imread(img_name)
    imgA = cv2.resize(imgA, (160, 160))
    imgA = transform(imgA)
    imgA = imgA.to(device)
    imgA = imgA.unsqueeze(0)
    
    output = model(imgA)
    output = torch.sigmoid(output)
    output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)

    print(output_np.shape)   #(1, 2, 160, 160)
    output_np = np.argmin(output_np, axis=1)
    print(output_np.shape)  #(1,160, 160)

    plt.subplot(1, 2, 1)
    #plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
    #plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(output_np[0, ...]), 'gray')
    plt.pause(3)
