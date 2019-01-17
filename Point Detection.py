
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from copy import deepcopy


# In[2]:


img= cv2.imread('/Users/krishna/Downloads/point.jpg',0)
b=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]            


# In[3]:


def pointdetection(img,kernel):
    threshold=281
    t1=284
    list=[]
    image = deepcopy(img)
    row= img.shape[0]
    col= img.shape[1]
    sum=0
    for i in range(1, row-1):
            for j in range(1,col-1):
                sum=(kernel[0][0]*img[i-1][j-1] + kernel[0][1]*img[i-1][j] + kernel[0][2]*img[i-1][j+1] + kernel[1][0]*img[i][j-1] + kernel[1][1]*img[i][j] + kernel[1][2]*img[i][j+1] + kernel[2][0]*img[i+1][j-1] + kernel[2][1]*img[i+1][j] + kernel[2][2]*img[i+1][j+1])
                
                if sum>threshold and sum<t1:
                    image[i][j]=255
                    list.append((i,j))
                else:
                    image[i][j]=0
    x=list[0][0]
    y=list[0][1]
    coor=str(x)+','+str(y)
    cv2.circle(image,(y,x),10,(255,255,255),1)
    cv2.putText(image,coor,(y-60,x-20),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)                      
    return image


# In[4]:


result=pointdetection(img,b)
cv2.imwrite('/Users/krishna/Desktop/project3_cvip/pointdetection.jpg',result)
#cv2.namedWindow('pointdetection', cv2.WINDOW_NORMAL)
#cv2.imshow('pointdetection', result)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#plt.imshow(result)

