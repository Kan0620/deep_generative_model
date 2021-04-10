#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 16:18:57 2021

@author: nakaharakan
"""

from sklearn.datasets import fetch_openml
import torch
from torch import tensor
from torch.utils.data import TensorDataset,DataLoader
from torch.nn import functional as F,Linear,Module
from torch.optim import Adam
import matplotlib.pyplot as plt
import cv2

class VAE(Module):
    
    def __init__(self):
        
        super(VAE,self).__init__()
        self.enc_fc=Linear(784,200)
        self.enc_mean=Linear(200,10) 
        self.enc_var=Linear(200,10)
        self.dec_fc1=Linear(10,200)
        self.dec_fc2=Linear(200,784)
        self.optim=Adam(self.parameters(),lr=0.01)
        self.batch_size=8
    
    def forward(self,x):
        
        h1=F.relu(self.enc_fc(x))
        
        mean=self.enc_mean(h1)#平均μの出力
        
        var=self.enc_var(h1).exp()#分散σ^2の出力
        
        eps=torch.randn(mean.size())
        
        z=mean+torch.sqrt(var)*eps#正規分布でサンプリング
        
        h2=F.relu(self.dec_fc1(z))
        
        y=torch.sigmoid(self.dec_fc2(h2))
        
        
        return mean,var,y
    
    def fit(self,x):
        
        t=tensor(x,dtype=float)
        x=tensor(x,dtype=float,requires_grad=True)
        
        fit_set=TensorDataset(x,t)
        fit_loader=DataLoader(fit_set,batch_size=512,shuffle=True)
        
        self.train()
        
        train_loss=0
        
        print(len(fit_loader))
        
        count=0
        
        for data,targets in fit_loader:
            
            print(count)
            
            self.optim.zero_grad()
            
            mean,var,y=self(data.float())
            
            loss_1=-(targets*(y+1e-8).log()+(1-targets)*(1-y+1e-8)).mean()#再構成誤差
            
            loss_2=(-1/2*(1+var.log()-mean**2-var).sum(dim=1)).mean()#正規化誤差
            
            loss=loss_1+loss_2
            
            train_loss+=loss
            
            loss.backward()
            
            self.optim.step()
            
            count+=1
            
        print(train_loss)
            
    def generate(self,epoch):
    
        
        self.eval()
        
        
        
        z=torch.randn(self.batch_size,10)
        
        h2=F.relu(self.dec_fc1(z))
        
        y=torch.sigmoid(self.dec_fc2(h2))
        
        imgs=y.view(-1,28,28)
        
        imgs=imgs.squeeze().detach().numpy()
        
        return imgs
            
            
        
def main():
    
    data=fetch_openml('mnist_784').data/255
    
    model=VAE()
    
    for epoch in range(10):
        
        model.fit(data)
        imgs=model.generate(epoch)
        
        for i,img in enumerate(imgs):
            
            cv2.imwrite('epoch:'+str(epoch)+'_index:'+str(i)+'.jpg',img*255)
            print(img)
            
            
if __name__=='__main__':
    
    main()













