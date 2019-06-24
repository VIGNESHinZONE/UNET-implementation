#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
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
class nuclei_dataset(Dataset):
    def __init__(self,main_path,image_size=128):
        self.all_ids=os.listdir(main_path)
        self.images=[]
        self.masks=[]
        for ids in self.all_ids:
            image_path=os.path.join(main_path,ids,'images',ids)+'.png'
            mask_path=os.path.join(main_path,ids,'masks/')
            all_masks=os.listdir(mask_path)
            
            image = cv2.imread(image_path,1)
            image = cv2.resize(image,(image_size,image_size))
            image=image/255.0
            
            mask = np.zeros((image_size,image_size,1))
            for name in all_masks:
                _mask_path_=mask_path+name
                mask_image = cv2.imread(_mask_path_,-1)
                mask_image = cv2.resize(mask_image,(image_size,image_size))
                mask_image = np.expand_dims(mask_image, axis=-1)
                mask=np.maximum(mask,mask_image)
                
            mask=mask/255.0
            
            self.images.append(image)
            self.masks.append(mask)
            
        self.len=len(self.all_ids)
    def __getitem__(self, index):
        single_image=self.images[index]
        single_mask=self.masks[index]
        
        img=torch.FloatTensor(single_image)
        msk=torch.FloatTensor(single_mask)
        msk=msk.permute(2,0,1)
        img=img.permute(2,0,1)
        
        return (img,msk)
    def __len__(self):
        return self.len

        


# In[ ]:




