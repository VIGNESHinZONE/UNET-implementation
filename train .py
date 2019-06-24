#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Dataset import nuclei_dataset
from Unetmodel import unet
import torchvision.transforms as transforms
import os
import sys
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
seed = 2019
random.seed = seed
np.random.seed = seed


# In[2]:


path='C:/Users/VIGNESH/Desktop/project/unet/dataset/1/'
custom = nuclei_dataset(path)


# In[3]:


data_here=DataLoader(dataset=custom,batch_size=8,shuffle=False)


# In[ ]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=unet(3,1).to(device)


# In[ ]:


num_epochs = 20
learning_rate = 0.001
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(data_here)
curr_lr = learning_rate


# In[ ]:



for epoch in range(num_epochs):
    for i, (images, masks) in enumerate(data_here):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 50 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# In[ ]:


torch.save(model.state_dict(), 'unet.ckpt')

