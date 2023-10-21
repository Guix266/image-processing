# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:58:33 2020

@author: guix
TP 4: image restoration - denoising
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

import img_funct

# =============================================================================
# 4.1 Generation of random noise
# =============================================================================

def rand_noise(size, random):
    """ Return a blured image of the good size and the random distribution
        the values are between 0 and 255
    ---
    ARG : Random  : Uniforme = 'uni'
                    Gaussian = 'gauss'
                    Salt and pepper = 'S&P'
                    exponential = 'exp' """
    a = 0.1
    b = 0.9
    image = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            if random == 'uni':
                image[i,j] = np.random.uniform()
            elif random == 'gauss':
                c = -1
                while c<0 or c >6:
                    c = np.random.normal(3,1)
                image[i,j] = c
            elif random == 'S&P':
                R = np.random.uniform()
                if (R > a) and (R <= b) :
                    image[i,j] = 0.5
                elif R > b :
                    image[i,j] = 1
            elif random == 'exp' :
                image[i,j] = - np.log( 1 - np.random.uniform() )
    
    M = np.max(image)
    image = image / M * 255
    return (image)

noise1 = rand_noise([32,32], 'uni')
noise2 = rand_noise([32,32], 'gauss')
noise3 = rand_noise([32,32], 'S&P') 
noise4 = rand_noise([32,32], 'exp') 

fig = img_funct.plot_im_hist([noise1, noise2, noise3, noise4], ["Uniform noise", "Gaussian noise", "Salt and pepper noise", "Exponential noise"], binary = True)
fig.suptitle("Noise", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP4_noise_generation.jpg"))


# =============================================================================
# 4.2 Noise estimation
# =============================================================================
#%%

im1 = img_funct.load_im("jambe.png")
im1 = im1[:,:,0] * 255

# Region of interrest
ROI = im1[ 200:270 , 400:470]

# Add exponential noise
ROI_2 = ROI + rand_noise([70,70], 'exp') / 2
ROI_2 = img_funct.bit_image(ROI_2)

# Add gaussian noise
ROI_3 = ROI + rand_noise([70,70], 'gauss')  / 2
ROI_3 = img_funct.bit_image(ROI_3)

fig = img_funct.plot_im_hist([im1, ROI, ROI_2, ROI_3], ["jambe", "ROI", "ROI + exponential noise", "ROI + gaussian noise"], binary = True)
fig.suptitle("Noise", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP4_ROI.jpg"))

# =============================================================================
# 4.3 Image restoration by spatial filtering 
# =============================================================================
#%%

im1_noisy = im1 + rand_noise(np.shape(im1), 'gauss')/2
im1_noisy = img_funct.bit_image(im1_noisy)

# Mean
mean = img_funct.bit_image(scipy.ndimage.uniform_filter(im1_noisy, 5))

# Median
median = img_funct.bit_image(scipy.ndimage.median_filter(im1_noisy, 5))

# Min
mini = img_funct.bit_image(scipy.ndimage.minimum_filter(im1_noisy, 5))

# Max
maxi = img_funct.bit_image(scipy.ndimage.maximum_filter(im1_noisy, 5))


fig = img_funct.plot_im_hist([im1, im1_noisy, mean, median, mini, maxi], ["initial", "noisy", "mean", "median", "mini", "maxi"], binary = True)
fig.suptitle("Restauration", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP4_restauration.jpg"))

# Algo amélioré
#%%

def Min(image, i, j, S):
    m, n = np.shape(image)
    li = []
    for k in range(i-S, i+S+1):
        for l in range(j-S, j+S+1):
            if (k >= 0) and (k < m) and (l >= 0) and (l < n):
                li.append(image[k, l])
    return min(li)
    
    
    
def Max(image, i, j, S):
    m, n = np.shape(image)
    li = []
    for k in range(i-S, i+S+1):
        for l in range(j-S, j+S+1):
            if (k >= 0) and (k < m) and (l >= 0) and (l < n):
                li.append(image[k, l])
    return max(li)

def Med(image, i, j, S):
    m, n = np.shape(image)
    li = []
    for k in range(i-S, i+S+1):
        for l in range(j-S, j+S+1):
            if (k >= 0) and (k < m) and (l >= 0) and (l < n):
                li.append(image[k, l])
    return np.median(np.array(li))


def isMedImpulseNoise(image, i, j, S):
    med = Med(image, i, j, S)
    if med == Max(image, i, j, S) or med == Min(image, i, j, S) :
        return True
    else:
        return False

def adaptative_median_filter(image, nb_neighbour):
    m, n = np.shape(image)
    target = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            S = 1
            while isMedImpulseNoise(image, i, j, S) and S <= nb_neighbour :
                S = S + 1
                med = Med(image, i, j, S)
            if image[i,j] == Min(image, i, j, S) or image[i, j] == Max(image, i, j, S) or S == nb_neighbour:
                target[i, j] = Med(image, i, j, S)
            else:
                target[i,j] = image[i, j]
    return(target)    
                
    
median2 = adaptative_median_filter(im1, 5)

fig = img_funct.plot_im_hist([im1, median, median2, im1_noisy], ["initial", "median", "median +", "noisy"], binary = True)
fig.suptitle("Restauration 2", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP4_restauration2.jpg"))



