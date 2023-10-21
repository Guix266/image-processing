# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:19:08 2020

@author: guix
TP 5: image registration (25)
"""

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.image as im
import random as r
import os, sys
import imageio
import scipy
from scipy import ndimage
import skimage.exposure
from PIL import Image 
import cv2

import img_funct

im1 = img_funct.load_im("BrainT1.bmp")
im2 = img_funct.load_im("BrainT1bis.bmp")

# =============================================================================
# 25.1 Transformation estimation
# =============================================================================
#%%


A_points = np.array ([[136, 100], [125, 153], [95, 157], [86, 99]])
B_points = np.array ([[137, 94] , [109, 141], [79, 135], [91, 76]])

im1b = im1.copy()
im2b = im2.copy()

S = 2
for p in A_points:
    for k in range(p[0]-S, p[0]+S+1):
        for l in range(p[1]-S, p[1]+S+1):
                im1b[l, k] = 255
for p in B_points:
    for k in range(p[0]-S, p[0]+S+1):
        for l in range(p[1]-S, p[1]+S+1):
                im2b[l, k] = 255


fig = img_funct.plot_im([im1b, im2b], ["Brain1", "Brain2"], binary = True)
fig.suptitle("Brain", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP5_brain.jpg"))


# =============================================================================
# 25.2 Rigid transformation
# =============================================================================
#%%

def rigid_transform(points_im1, points_im2):
    """ Return the translation and rotation matrix of a transformation 
    on this form        | R1 R2 T1 |
                    M = | R3 R4 T2 |                                      
                        | 0  0  1  |
    """
    
    # computes barycenters, and recenters the points
    m1 = np.mean(points_im1,0)
    m2 = np.mean(points_im2,0) ;
    data1_inv_shifted = points_im1-m1;
    data2_inv_shifted = points_im2-m2;

    # Evaluates SVD
    K = np.matmul(np.transpose(data2_inv_shifted ), data1_inv_shifted)
    U,S,V = np.linalg.svd(K)

    # Computes Rotation
    S = np.eye(2) ;
    S[1,1] = np.linalg.det(U) * np.linalg.det(V)
    R = np.matmul(U,S)
    R = np.matmul(R, np.transpose(V))

    # Computes Translation
    T = m2-np.matmul(R, m1)
    
    M = np.eye(3)
    M[0:2,0:2] = R
    M[0:2,2] = T
    print("Matrix transform M = ")
    print(M)
    return(M)

M = rigid_transform(A_points, B_points)

def superimpose(G1, G2):
    """
        3 superimpose 2 images, supposing they are grayscale images and of same shape
    """
    r , c=G1.shape
    S = np.zeros (( r , c ,3) )
    S [:,:,0] = np.maximum(G1-G2, 0)+G1
    S [:,:,1] = np.maximum(G2-G1, 0)+G2
    S [:,:,2] = (G1+G2) / 2
    S = 255 * S / np.max(S)
    S = S.astype( 'uint8' )
    # plt.imshow(S);
    # plt.show()
    return S




# Apply transformation on image
rows, cols= np.shape(im1)
dst = cv2.warpAffine(im1, M[0:2,:], ( cols , rows))
sup = superimpose(dst , im2)

fig = img_funct.plot_im([im1, im2, dst, sup], ["Brain1", "Brain2", "brain1 modified", "Superposition"], binary = True)
fig.suptitle("Rigid transform", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP5_rigid_transform.jpg"))

# =============================================================================
# 25.3 ICP-based registration
# =============================================================================
#%%

#take points randomly
A = np.random.randint(0, 221, size = [4,2])
B = np.random.randint(0, 221, size = [4,2])

M = rigid_transform(A, B)
rows, cols= np.shape(im1)
dst = cv2.warpAffine(im1, M[0:2,:], ( cols , rows))
sup = superimpose(dst , im2)
fig = img_funct.plot_im([im1, im2, dst, sup], ["Brain1", "Brain2", "brain1 modified", "Superposition"], binary = True)








