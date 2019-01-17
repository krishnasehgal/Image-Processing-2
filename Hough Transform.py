
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
from math import sqrt


# ###### Reference: https://github.com/alyssaq/hough_transform/blob/master/hough_transform.py

# In[2]:


def angled(image,accumulator, theta, r):
    
    idx1 = np.argsort(accumulator.ravel(), axis=None)[-25:]  
    idx2= np.argsort(accumulator.ravel(), axis=None)[-50:-21]
    idx= np.hstack((idx1,idx2))
    rho = r[idx // accumulator.shape[1]]
    theta_1 = theta[idx % accumulator.shape[1]]

    for r, t in zip(rho,theta_1):
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1,y1),(x2,y2),(255,0,0),2)
    plt.imshow(image)
    return image


# In[3]:


def vertical(image,accumulator, theta, r):
    idx = np.argsort(accumulator.ravel(), axis=None)[-25:-15]  #-50:-47
    rho = r[idx // accumulator.shape[1]]
    theta_1 = theta[idx % accumulator.shape[1]]

    for r, t in zip(rho,theta_1):
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(image, (x1,y1),(x2,y2),(0,0,255),2)
    plt.imshow(image)  
    return image
    


# In[6]:


image= cv2.imread("/Users/krishna/Downloads/original_imgs/hough.jpg",0)
image1=cv2.imread("/Users/krishna/Downloads/original_imgs/hough.jpg")
image2=cv2.imread("/Users/krishna/Downloads/original_imgs/hough.jpg")

x=cv2.Canny(image,100,250)
gradient=1
threshold=100.0

diagonal = int(round(math.sqrt((x.shape[0])**2 + (x.shape[1])**2)))

rho = np.linspace(-diagonal, diagonal, diagonal * 2)

theta = np.deg2rad(np.arange(0, 360.0, gradient))
theta_1 = len(theta)

accumulator = np.zeros((2 * diagonal, theta_1), dtype=np.uint32)
edges = x > threshold
idx_y, idx_x = np.nonzero(edges)

for i in range(len(idx_x)):
    x = idx_x[i]
    y = idx_y[i]

    for idx_t, angle in enumerate(theta):
        rho_1 = diagonal+ int(round(x * math.cos(angle) + y * math.sin(angle)))
        accumulator[rho_1, idx_t] += 1

ans1=angled(image1, accumulator, theta, rho) 
ans2= vertical(image2, accumulator, theta, rho)

cv2.imwrite("/Users/krishna/Desktop/project3_cvip/blue_lines.jpg", ans1)
cv2.imwrite("/Users/krishna/Desktop/project3_cvip/red_lines.jpg", ans2)

