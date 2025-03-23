#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv


# In[7]:


data_dir = '../_data/cifar-10-batches-py/'
fileList = os.listdir(data_dir)
fileList


# In[8]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[9]:


dict1 = unpickle(data_dir+fileList[1])
dict2 = unpickle(data_dir+fileList[2])
dict3 = unpickle(data_dir+fileList[3])
dict4 = unpickle(data_dir+fileList[4])
dict5 = unpickle(data_dir+fileList[5])
dict7 = unpickle(data_dir+fileList[7])


# In[10]:


dict1[b'data'].shape, dict1[b'filenames'][0], dict1[b'batch_label']


# In[11]:


len(dict1[b'data'][0][:1024])
len(dict1[b'data'][0][1024:2048])
len(dict1[b'data'][0][2048:])


# In[13]:


# for i in range(10000):
def inverse_image(dict, idx):    
    B = dict[b'data'][idx][:1024]
    G = dict[b'data'][idx][1024:2048]
    R = dict[b'data'][idx][2048:]
    img_rgb = cv2.merge((B,G,R))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_save =  img_gray.reshape(32,32)
    image_dir= data_dir+f"batch_{dict[b'batch_label'].decode()}/"
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    else:
        quit()
    cv2.imwrite(image_dir+dict[b'filenames'][idx].decode(), img_save)
    return img_gray


# In[14]:


def img2df(dict, img_gray, idx):
    arrays1 = np.insert(img_gray.reshape(-1), 0, dict[b'labels'][idx])
    with open(data_dir+dict1[b'batch_label'].decode()+'.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(arrays1)
    


# In[ ]:


for i in range(10000):
    a = inverse_image(dict7, i)
    img2df(dict7, a,i)


# In[ ]:


for i in range(10000):
    a = inverse_image(dict5, i)
    img2df(dict5, a,i)


# In[ ]:


for i in range(10000):
    a = inverse_image(dict4, i)
    img2df(dict4, a,i)


# In[ ]:


for i in range(10000):
    a = inverse_image(dict2, i)
    img2df(dict2, a,i)


# In[ ]:


for i in range(10000):
    a = inverse_image(dict1, i)
    img2df(dict1, a,i)


# In[ ]:




