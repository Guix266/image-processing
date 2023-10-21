# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:22:48 2020

@author: guix
TP3 : Fourier transform
"""

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.image as im
import os, sys
import imageio
import scipy
from scipy import ndimage
import skimage.exposure

import img_funct

# =============================================================================
# 3.1 Fourier transform
# =============================================================================
#%%
im1 = img_funct.load_im("cornee.png", grey = True)

# Fourier transform calcul
graph = np.fft.fftshift(np.fft.fft2(im1)) #fftshift, put 0 frequency on the center of the image

fig = img_funct.plot_pectrumPhase(im1, graph)

fig.suptitle("Fourier Transform : Cornee", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_cornee.jpg"))


# =============================================================================
# 3.2 Inverse fourier transform
# =============================================================================
#%%

im2 = img_funct.load_im("lena256.bmp", grey = True)
image = im2

# Fourier transform
four = np.fft.fftshift(np.fft.fft2(image))
phase = np.angle(four)
ampli = abs(four)

# Inverse of the fourier transform
inv1 = np.real(np.fft.ifft2(np.fft.ifftshift(four)))

fig = img_funct.plot_im( [image, inv1] , ["Original", "After fourier"], binary = True)
fig.suptitle("Fourier inverse", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_inverse.jpg")) 


# Apply inverse on the phase / amplitude
invampli = np.real(np.fft.ifft2(np.fft.ifftshift(ampli)))
complex_phase = np.exp(1j*phase)
invphase = np.real(np.fft.ifft2(np.fft.ifftshift(complex_phase)))

fig = img_funct.plot_im( [inv1, invampli, invphase] , ["Reconstruction totale", "Reconstruction amplitude", "Reconstruction phase"], binary = True)
fig.suptitle("Reconstructions", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_reconstruction.jpg")) 


# =============================================================================
# 3.3 Low-pass / high pass filters
# =============================================================================
#%%

def low_pass(fourier_2D, size):
    """ Return the fourier transform after that the low pass filter has been applied
        - Size : [0,1] represent the spectral force
    """
    shape = np.shape(fourier_2D)
    mask = np.zeros(shape)
    
    mx = shape[0]/2 
    my = shape[1]/2 
    
    x_lim = shape[0]/2 * size
    y_lim = shape[1]/2 * size
    
    mask[int(mx-x_lim) : int(mx+x_lim), int(my-y_lim) : int(my+y_lim)] = 1
    
    image = mask*fourier_2D
    return(image)
    
def high_pass(fourier_2D, size):
    """ Return the fourier transform after that the high pass filter has been applied
        - Size : [0,1] represent the spectral force
    """
    shape = np.shape(fourier_2D)
    mask = np.ones(shape)
    
    mx = shape[0]/2 
    my = shape[1]/2 
    x_lim = shape[0]/2 * size
    y_lim = shape[1]/2 * size
    
    mask[int(mx-x_lim) : int(mx+x_lim), int(my-y_lim) : int(my+y_lim)] = 0
    
    image = mask*fourier_2D
    return(image) 
    

# Load the image
image = im2
# Fourier transform
four = np.fft.fftshift(np.fft.fft2(image))

########################################################
# Effect of the low pass
low_four = low_pass(four, 0.5)
img_funct.plot_pectrumPhase(image, low_four)
low_inv = np.real(np.fft.ifft2(np.fft.ifftshift(low_four)))

# Effect of the high pass
high_four = high_pass(four, 0.5)
img_funct.plot_pectrumPhase(image, high_four)
high_inv= np.real(np.fft.ifft2(np.fft.ifftshift(high_four)))


fig = img_funct.plot_im( [image, low_inv, high_inv] , ["image", "Fourier low pass", "Fourier high pass"], binary = True)
fig.suptitle("Fouriers filters", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_filters.jpg")) 

#########################################################
# Influence of the size

low_str = [image]
for s in [0.1, 0.2, 0.4, 0.5]:
    low_four = low_pass(four, s)
    # img_funct.plot_pectrumPhase(image, low_four)
    low_im = np.real(np.fft.ifft2(np.fft.ifftshift(low_four)))
    low_str.append(low_im)

fig = img_funct.plot_im( low_str , ["Normale", "0.1", "0.2", "0.4", "0.5"], binary = True)
fig.suptitle("Fouriers lowfilters", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_low.jpg")) 


high_str = [image]
for s in [0.1, 0.2, 0.4, 0.5]:
    high_four = high_pass(four, s)
    # img_funct.plot_pectrumPhase(image, low_four)
    high_im = np.real(np.fft.ifft2(np.fft.ifftshift(high_four)))
    high_str.append(high_im)
    
fig = img_funct.plot_im( high_str , ["Normale", "0.1", "0.2", "0.4", "0.5"], binary = True)
fig.suptitle("Fouriers highfilters", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_fourier_high.jpg")) 


# =============================================================================
# 3.4 Application: evaluation of cellular density
# =============================================================================
#%%

im1 = img_funct.load_im("cornee.png", grey = True)
graph = np.fft.fftshift(np.fft.fft2(im1))
fig = img_funct.plot_pectrumPhase(im1, graph)
# We observe a vertival line that indicate a patern in the cell layers

# Apply a gaussian filter on the amplitude
ampli = abs(graph)
ampli = ndimage.gaussian_filter(ampli, 5)
ampli = np.log(1 + abs(ampli))

plt.subplot(121)
plt.imshow(ampli, cmap = "gray")
plt.title("amplitude after gauss", fontsize=24)

plt.subplot(122)
plt.plot(ampli[:,int(np.shape(ampli)[1]/2)])
plt.title("Horizontal trench", fontsize=24)
plt.show()

fig.suptitle("Amplitude", fontsize = 26, weight = 'bold')
fig.savefig(os.path.join("processed_images","TP3_cell_amplitude.jpg")) 





