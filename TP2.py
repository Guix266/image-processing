# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:17:52 2020

@author: guix
TP2 : Image Enhancement
"""

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.image as im
import os, sys
import imageio
import scipy
import skimage.exposure

import img_funct

# =============================================================================
#  1.1 Intensity transformation
# =============================================================================
#%%
im1 = img_funct.load_im("osteoblaste.jpg", grey = True)
im1_norm = im1 / np.max(im1) 

# Gamma correction
im_gamma = [im1]
for g in [0.1, 0.5, 1, 1.5, 2]:
    im_gamma.append(skimage.exposure.adjust_gamma(im1_norm, gamma = g) *255)

fig = img_funct.plot_im_hist( im_gamma , ["original", "gamma = 0.1", "gamma = 0.5", "gamma = 1", "gamma = 1.5", "gamma = 2"], binary = True)
fig.suptitle("Gamma Correction", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_gamma.jpg"))

# Constrast Stretching
#%%

def Stretching(image, E):
    """Return the stretched image with a E factor"""
    epsilon = sys.float_info.epsilon
    mean = np.mean(image)
    image = 1/( 1 + (mean/(epsilon + image))**E)
    return(image)

im_stretch = [im1]
for e in [10, 20, 100, 1000]:
    im_stretch.append(Stretching(im1_norm, e) * 255)

fig = img_funct.plot_im_hist( im_stretch , ["original", "E = 10", "E = 20", "E = 100", "E = 1000"], binary =  True)
fig.suptitle("Contrast Stretching", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_stretching.jpg"))  


# =============================================================================
# 2.2 Histogram equalization
# =============================================================================
#%%

# Equalized function
im_equa = skimage.exposure.equalize_hist(im1) * 255

# Equalized created
def equalize(image): 
    hist , bins=np.histogram(image.flatten(),256, range = (0,255)) 
    cdf = hist.cumsum()
    cdf = (cdf/cdf[- 1])
    return (cdf[image])

im_equa2 = equalize(im1) *255

fig = img_funct.plot_im_hist( [im1, im_equa, im_equa2] , ["original", "Equalized", "Equalized created"], binary = True)
fig.suptitle("Histogram Equalization", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_equalize.jpg"))  

# =============================================================================
# 2.3 Histogram matching
# =============================================================================
#%%

im2 = img_funct.load_im("phobos.jpg", grey = True)

# Plot the image
fig = img_funct.plot_hist( [im2] , ["phobos.png"])
fig.suptitle("phobos.jpg", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_phobos.jpg"))  

# PLot the equalized one
im_equa = skimage.exposure.equalize_hist(im2) * 255
fig = img_funct.plot_im_hist( [im2, im_equa] , ["original", "Equalized"], binary = True)
fig.suptitle("Phobos", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_phobos_equalize.jpg"))  
#%%

# bi-modal histogram
def hist_matching(image, cdf_dest):
    """ Histogram matching of image I , with cumulative histogram cdf_dest
    This should be normalized, between 0 and 1.
    This version uses interpolation """
    
    imhist , bins = np.histogram(image.flatten () , len( cdf_dest ) , density=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = ( cdf / cdf [ - 1]) #normalize between 0 and 1
    
    # 1rst : histogram equalization
    im2 = np.interp(image.flatten(), bins[:-1], cdf)
    # 2nd: reverse function
    im3 = np. interp (im2, cdf_dest , bins[:-1])
    # reshape into image
    imres = im3.reshape(image.shape)
    return (imres)

def func(x, sigma, mu):
    x = 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x - mu)**2 / (2 * sigma**2))
    return(x)

cdf_dest = []
x = []
for i in range(0,256):
    x.append(i)
    cdf_dest.append(func(i, 30, 25))

# plt.plot(x,cdf_dest)

imres = hist_matching(im2, cdf_dest)*255
                                          
fig = img_funct.plot_im_hist( [im2, imres] , ["original", "Matched patern"], binary = True)
fig.suptitle("Phobos_match", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP2_phobos_matched.jpg"))  

