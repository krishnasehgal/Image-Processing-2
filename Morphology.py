
# coding: utf-8

# In[10]:


import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from copy import deepcopy


# In[11]:


img= cv2.imread('/Users/krishna/Downloads/original_imgs/noise.jpg',0)
b=[[0,255,0],[255,255,255],[0,255,0]]            


# In[12]:


def erosion(img,kernel):
    image = deepcopy(img)
    row= img.shape[0]
    col= img.shape[1]
    t=255*255*5
    for i in range(1, row-1):
            for j in range(1,col-1):
                if ((kernel[0][0]*img[i-1][j-1] + kernel[0][1]*img[i-1][j] + kernel[0][2]*img[i-1][j+1] + kernel[1][0]*img[i][j-1] + kernel[1][1]*img[i][j] + kernel[1][2]*img[i][j+1] + kernel[2][0]*img[i+1][j-1] + kernel[2][1]*img[i+1][j] + kernel[2][2]*img[i+1][j+1])==t):
                    image[i][j]=255
                else:
                    image[i][j]=0            
    return image

def dilation(image,kernel):
    img = deepcopy(image)
    row= img.shape[0]
    col= img.shape[1]
    t=255*255*5
    for i in range(1, row-1):
            for j in range(1,col-1):
                if (b[0][0]*img[i-1][j-1] + b[0][1]*img[i-1][j] + b[0][2]*img[i-1][j+1] + b[1][0]*img[i][j-1] + b[1][1]*img[i][j] + b[1][2]*img[i][j+1] + b[2][0]*img[i+1][j-1] + b[2][1]*img[i+1][j] + b[2][2]*img[i+1][j+1]) > 0:
                    image[i][j]=255
                else:
                    image[i][j]=0   
    return image


# In[ ]:


result = erosion(img,b)
result = dilation(result, b)
result = dilation(result, b)
result = erosion(result,b)

result1=dilation(img,b)
result1=erosion(result1, b)
result1=erosion(result1, b)
result1=dilation(result1,b)

result2=erosion(result,b)
result2=result-result2

result3=erosion(result1,b)
result3=result1-result3


# In[ ]:


cv2.imwrite('/Users/krishna/Desktop/project3_cvip/res_noise1.jpg',result)
cv2.imwrite('/Users/krishna/Desktop/project3_cvip/res_noise2.jpg',result1)
cv2.imwrite('/Users/krishna/Desktop/project3_cvip/res_bound1.jpg',result2)
cv2.imwrite('/Users/krishna/Desktop/project3_cvip/res_bound2.jpg',result3)

