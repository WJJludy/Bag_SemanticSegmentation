'''
train.py
训练模型
'''
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#import visdom
from dataset import test_dataloader, train_dataloader
#from model import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from model import FCNs,VGGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#device=torch.device("cpu")      #使用cpu
#device=torch.device("cuda")     #使用GPU。
def train(epo_num=50, show_vgg_params=False):
    #vis = visdom.Visdom()   #pytorch中的可视化工具
    
    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    #将模型加载到指定设备上
    fcn_model = fcn_model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)

    all_train_iter_loss = []
    all_test_iter_loss = []

    #计算时间
    prev_time = datetime.now()
    for epo in range(epo_num):
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_dataloader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])
            bag = bag.to(device)
            bag_msk = bag_msk.to(device)

            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            # print(output)
            # print(bag_msk)
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
            output_np = np.argmin(output_np, axis=1)
            bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
            bag_msk_np = np.argmin(bag_msk_np, axis=1)

        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (bag, bag_msk) in enumerate(test_dataloader):
                bag = bag.to(device)
                bag_msk = bag_msk.to(device)
                optimizer.zero_grad()
                output = fcn_model(bag)
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
                loss = criterion(output, bag_msk)  #预测和原标签图的差
                iter_loss = loss.item()   #item得到一个元素张量里面的元素值，一般用于返回loss，acc
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmin(output_np, axis=1)
                bag_msk_np = bag_msk.cpu().detach().numpy().copy() # bag_msk_np.shape = (4, 2, 160, 160) 
                bag_msk_np = np.argmin(bag_msk_np, axis=1)

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
 

        print('epoch train loss = %f, epoch test loss = %f, %s'
                %(train_loss/len(train_dataloader), test_loss/len(test_dataloader), time_str))


        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, '/Bags/results/fcn_model_{}.pt'.format(epo))
            print('fcn_model_{}.pt'.format(epo))


if __name__ == "__main__":
    train(epo_num=100, show_vgg_params=False)
