# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:27:25 2020

@author: guix
TP 6: Histogram-based image segmentation (19)
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
from skimage import filters
import sklearn.cluster
from mpl_toolkits.mplot3d import Axes3D
import cv2

import img_funct

im1 = img_funct.load_im("cells.bmp")

# =============================================================================
# 19.1 Transformation estimation
# =============================================================================
#%%

def thresh(image, th):
    n, m = np.shape(image)
    thr = np.zeros([n, m])
    for i in range(n):
        for j in range(m):
            if image[i,j]>=th:
                thr[i,j] = 1
    return(thr)

# Threeshold of t = 105
thr1 = thresh(im1, 90)
thr2 = thresh(im1, 100)
thr3 = thresh(im1, 105)
thr4 = thresh(im1, 110)

fig = img_funct.plot_im_hist([im1, thr1, thr2, thr3, thr4],["cells", "threshold t=90", "threshold t=100", "threshold t=105", "threshold t=110"], binary = True)
fig.suptitle("threshold", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP6_threshold.jpg"))

# =============================================================================
# 19.2 k-means clustering
# =============================================================================
#%%

#Algo valeur mediane

# OSU
th = filters.threshold_otsu(im1)


# =============================================================================
# 19.3 k-means clustering k =3
# =============================================================================
#%%

# Génération points aléatoires autour de P(x,y)
def generation(n, x, y) :
    Y = np.random.randn(n, 2) + np.array([[ x, y ]])
    return(Y)

points1=generation (100, 0, 0)
points2=generation (100, 3,4)
points3=generation (100, - 5, - 3)

#Plot points
p = np.concatenate((points1,points2,points3))
np.random.shuffle(p)
plt.plot(p[:,0], p[:,1],'bo')
plt.figure()

#K means
kmean = sklearn.cluster.KMeans(n_clusters=3).fit(p)
label = kmean.labels_
point1 = [[],[]]
point2 = [[],[]]
point3 = [[],[]]
for i in range(np.shape(p)[0]):
    if label[i] == 0:
        point1[0].append(p[i,0])
        point1[1].append(p[i,1])
    elif label[i] == 1:
        point2[0].append(p[i,0])
        point2[1].append(p[i,1])
    else:
        point3[0].append(p[i,0])
        point3[1].append(p[i,1])

plt.plot(point1[0], point1[1],'bo')
plt.plot(point2[0], point2[1],'ro')
plt.plot(point3[0], point3[1],'yo')
plt.show()

# =============================================================================
# 19.4 Color image segmentation usingK-means: k = 3 in 3D
# =============================================================================
#%%

im2 = img_funct.load_im("Tv16.png")

# Flatten
n, m , _ = np.shape(im2)
flat = np.reshape(im2, [n*m,3])

# Visualise histogram 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# np.random.shuffle(flat)
# ax.scatter3D(flat[:200,0], flat[:200,1], flat[:200,2])
# plt.show()

# KMean methode
kmean2 = sklearn.cluster.KMeans(n_clusters=3).fit(flat)
label = kmean2.labels_

segmentation = 70 * np.reshape(label , (n, m))

fig = img_funct.plot_im([im2, segmentation],["cells", "Segmented image"])
fig.suptitle("threshold", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP6_threshold_colored.jpg"))




