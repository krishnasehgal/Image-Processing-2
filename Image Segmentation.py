
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
from copy import deepcopy


# In[ ]:


img= cv2.imread('/Users/krishna/Downloads/original_imgs/segment.jpg',0)
img=np.asarray(img)
row= img.shape[0]
col= img.shape[1]
list_=np.zeros((256,), dtype=int)
array=np.zeros((row,col), dtype=int)
count=0
threshold=205
for i in range(0,row-1):
    for j in range(0,col-1):
        list_[img[i][j]]+=1
l = [i for i in range(256)]

for x in range(0,row-1):
    for y in range(0,col-1):
        if img[x][y]>threshold:
            array[x][y]=255
        else:
            array[x][y]=0
            
array=array.astype(np.uint8).copy()            
plt.fill_between(l,list_,color='red')
plt.plot(l,list_,'-',color='red')
plt.grid(True)
plt.show()


#plt.imshow(array)
#cv2.namedWindow('image_segmentation', cv2.WINDOW_NORMAL)
#cv2.imshow('image_segmentation', array)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('/Users/krishna/Desktop/project3_cvip/image_segmentation.jpg',array)

