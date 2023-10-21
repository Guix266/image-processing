# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:16:18 2020

@author: guillaume.desermeaux
TP1 : Introduction to image processing
"""

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.image as im
import os, sys
import imageio
import scipy.ndimage

import img_funct

# =============================================================================
#  1.1
# =============================================================================
#%%
# Load the 3 images
im1 = img_funct.load_im("retine.png")
im2 = img_funct.load_im("muscle.jpg")
im3 = img_funct.load_im("cellules_cornee.jpg")

# Plot the 3 images
img_funct.plot_im([im1, im2, im3], ["retine.png", "muscle.jpg", "cellules_cornee.jpg"])

# Print the imformation of the images
print( "Shapes of the images:", np.shape(im1), np.shape(im2), np.shape(im3))
print( "Class of the images:", type(im1), type(im2), type(im3))

# Color components (green red blue)
img_funct.plot_im([im1[:,:,0], im1[:,:,1], im1[:,:,2]], ["Green component", "Red component", "Blue component"], binary = True)


# Difference between formats = loss of quality during the compression
# Write JPEG with different quality
for q in [1,25,50,100]:
    imageio.imwrite(os.path.join("processed_images","TP1_jpeg_"+str(q)+".jpeg"), im1, quality = q)

# =============================================================================
# 1.2
# =============================================================================
#%%

# Quantization of a grey image (muscle)
img = im2
quant_im =[img]
for i in range(1,9):
    quant_im.append(img//(2**i)*(2**i))
fig = img_funct.plot_im(quant_im,["q256", "q128", "q64", "q32", "q16", "q8", "q4", "q2", "q1"], binary = True)
fig.suptitle("Quantization")
fig.savefig(os.path.join("processed_images","TP1_quantization.jpg"))

# Computation of the corresponding histograms
fig = img_funct.plot_hist(quant_im ,["q256", "q128", "q64", "q32", "q16", "q8", "q4", "q2", "q1"])
fig.suptitle("Quantization histograms")
fig.savefig(os.path.join("processed_images","TP1_quantization_hist.jpg"))

# =============================================================================
# 1.3
# =============================================================================
#%%

# Computation of the corresponding histograms
fig = img_funct.plot_hist(quant_im ,["q256", "q128", "q64", "q32", "q16", "q8", "q4", "q2", "q1"])
fig.suptitle("Quantization histograms")
fig.savefig(os.path.join("processed_images","TP1_quantization_hist.jpg"))

# =============================================================================
# 1.4 Linear mapping of the image intensities
# =============================================================================
#%%

im = im3
mini = np.min(im)
maxi = np.max(im)
print("Le min est ",mini,"\nLe max est ", maxi )

a = 255 / (maxi-mini)
b = - (mini*a)
map_im = a*im + b

fig1 = img_funct.plot_im([im, map_im], ["Image", "Mapped image" ], binary = True)
fig2 = img_funct.plot_hist([im, map_im], ["Image", "Mapped image" ], 100)
fig1.suptitle("Linear mapping of the image intensities")
fig2.suptitle("Linear mapping of the image intensities")
fig1.savefig(os.path.join("processed_images","TP1_linmap.jpg"))
fig2.savefig(os.path.join("processed_images","TP1_linmap_hist.jpg"))

# =============================================================================
# 1.5 Aliasing effect
# =============================================================================
#%%

# aliasing effect (Moire)
def circle ( fs , f) :
    """Generates an image with aliasing effect
    # fs : sample frequency (fréquance d'échantillonage)
    # f : signal frequency"""
    t = np.arange (0,1,1./ fs)
    ti , tj = np.meshgrid(t,t)
    C = np.sin(2*np.pi*f*np.sqrt(ti**2+tj**2))
    print(t)
    print(ti, tj)
    print(C)
    return C

C1 = circle(100,20)
C2 = circle(100, 70)
fig = img_funct.plot_im([C1,C2],["f<fe/2", "f>fe/2"], binary = True)
fig.suptitle("Aliasing effect")
fig.savefig(os.path.join("processed_images","TP1_aliasing.jpg"))

#if f > fs/2, a signal of low frequency appears in the image

# =============================================================================
# 1.6 Low-past filtering
# =============================================================================
#%%

# Load the 2 images
im4 = img_funct.load_im("osteoblaste.jpg")
im5 = img_funct.load_im("blood.jpg")

img = im5
# Mean
mean3 = scipy.ndimage.uniform_filter(img, 3)
mean5 = scipy.ndimage.uniform_filter(img, 5)
# Median
median3 = scipy.ndimage.median_filter(img, 3)
median5 = scipy.ndimage.median_filter(img, 5)
# Min
mini = scipy.ndimage.minimum_filter(img, 3)
# Max
maxi = scipy.ndimage.maximum_filter(img, 3)
# Gaussian
gauss = scipy.ndimage.gaussian_filter(img, sigma = 1/2)

fig = img_funct.plot_im([mean3, mean5, median3, median5, mini, maxi, gauss],["mean 3", "mean 5", "median 3", "median 5", "mini", "maxi", "gauss"], binary = True)
fig.suptitle("low-past filtering")
fig.savefig(os.path.join("processed_images","TP1_lowpast.jpg"))
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_norm.jpg"), img)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_mean3.jpg"), mean3)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_mean5.jpg"), mean5)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_median3.jpg"), median3)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_median5.jpg"), median5)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_mini.jpg"), mini)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_maxi.jpg"), maxi)
imageio.imwrite(os.path.join("processed_images","TP1_lowfilt_gauss.jpg"), gauss)

# =============================================================================
# 1.7 High-past filtering
# =============================================================================
#%%

high_past = img - scipy.ndimage.uniform_filter(img, 25)
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
high_past2 = scipy.ndimage.convolve(img, kernel)

fig = img_funct.plot_im([img, high_past, high_past2], ["original", "high-past 1", "high-past 2"], binary = True)
fig.suptitle("high-past filtering")
fig.savefig(os.path.join("processed_images","TP1_highpast.jpg"))

# =============================================================================
# 1.8 Derivative filters
# =============================================================================
#%%

img = img_funct.load_im("osteoblaste.jpg", grey = True)
kernel = [np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
          np.array([[-1,-1,-1],[0,0,0],[1,1,1]]),
          np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
          np.array([[-1,-2,-1],[0,0,0],[1,2,1]])]

deriv = [img]
for i in kernel:
    deriv.append(scipy.ndimage.convolve(img, i))

fig = img_funct.plot_im(deriv, ["original", "Prewitt verticale", "prewitt horizontale", "Sobel verticale", "Sobel horizontale"], binary = True)
fig.suptitle("derivative filtering")
fig.savefig(os.path.join("processed_images","TP1_derivative.jpg"))

# =============================================================================
# 1.9 Enhancement fltering
# =============================================================================
#%%

# Enhence with laplacian filter
kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
liste = [img]
for alpha in [0, 0.5, 1, 3, 5, 10]:
    enhanced = alpha*img + scipy.ndimage.convolve(img, kernel)
    liste.append(enhanced)
    
fig = img_funct.plot_im(liste , ["original", "alpha = 0", "alpha = 0.5", "alpha =1", "alpha = 3", "alpha = 5", "alpha = 10"], binary = True)
fig.suptitle("Enhanced filtering")
fig.savefig(os.path.join("processed_images","TP1_enhanced.jpg"))















