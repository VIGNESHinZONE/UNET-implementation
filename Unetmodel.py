#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

def conv3x3(in_channels,out_channels,stride=1,padding=1):
    return nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=stride, padding=padding,bias= False)


class simpleblock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None,batchnorm=True):
        super(simpleblock,self).__init__()
        self.conv1=conv3x3(in_channels,out_channels)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2=conv3x3(out_channels,out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.bn=batchnorm
        self.maxpool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        residual=out
        out= self.maxpool(out)
        return residual,out
        
            
        
        
        
class simpleblockexpand(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,downsample=None,batchnorm=True,dropout=0):
        super(simpleblockexpand,self).__init__()
        self.conv1=conv3x3(in_channels,out_channels*2)
        self.bn1=nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2=conv3x3(out_channels*2,out_channels*2)
        self.bn2=nn.BatchNorm2d(out_channels*2)
        self.bn=batchnorm
        self.upsample=nn.ConvTranspose2d(out_channels*2, out_channels, 2, stride=2)
        
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        if self.bn:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        residual=out
        out= self.upsample(out)
        return residual,out
    
    
class unet(nn.Module):
    def __init__(self,in_channel,out_channel,batchnorm=True):
        super(unet,self).__init__()
        self.start=nn.Sequential(
            nn.Conv2d(in_channel, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.startpool=nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1=simpleblock(16,32)
        self.layer2=simpleblock(32,64)
        self.layer3=simpleblock(64,128)
        self.middle=nn.Sequential(
            nn.Conv2d(128,256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
        )
        self.firstupsample=nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.layer4=simpleblockexpand(256,64)
        self.layer5=simpleblockexpand(128,32)
        self.layer6=simpleblockexpand(64,16)
        self.end=nn.Sequential(
            nn.Conv2d(32,16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,1,1, padding=0),
        )
    def forward(self,x):
        c1=self.start(x)
        p1=self.startpool(c1)
        c2,p2=self.layer1(p1)
        c3,p3=self.layer2(p2)
        c4,p4=self.layer3(p3)
        c5=self.middle(p4)
        u6=self.firstupsample(c5)
        u6=torch.cat((u6,c4),1)
        c6,u7=self.layer4(u6)
        u7=torch.cat((u7,c3),1)
        c7,u8=self.layer5(u7)
        u8=torch.cat((u8,c2),1)
        c8,u9=self.layer6(u8)
        u9=torch.cat((u9,c1),1)
        out=self.end(u9)
        return out
        
        
        
        


# In[ ]:




