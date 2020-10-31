# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 04:53:24 2020

@author: MY PC
"""

import cv2
# import numpy as np
import matplotlib.pyplot as plt

# ############################### DEEP LEARNING MODEL TAINING #########################



# ############################### OPENCV ##############################################
# train_images=[] #Get 10 images for training
# train_images_vectors_dict={}

# frontal_face_xml='E:/AI and ML/interview and projects/projects/Face Recognition/haarcascade_frontalface_default.xml'

# for img in train_images:
    
#     ###Face Alignment
    
    
#     ### WARP Images
    
    
    
#     ### Detect Face
    
#     ### extract features
    
#     ### store in dict
#     pass


# ##################################################################

##### Load Image
img=cv2.imread('tiltedhead.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)

###### Face Detection 

