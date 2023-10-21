# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:34:55 2020

Functions to use in TB1_image processing

@author: guix
"""
# =============================================================================
# TP1
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import scipy
import imageio
from PIL import Image 
import os

def load_im(image, folder = "", grey = False):
    """Load an image and stock it (add folder inside of the 'images' folder)"""
    folder = os.path.join("images",folder)
    image = img.imread(os.path.join(folder,image))
    # print(np.shape(image))
    if grey:
        print("[INFO] : Grey image imported")
        if (len(np.shape(image)) >= 3):
            print("[INFO] : Image converted to grey scale")
            image = np.dot(image[...,:3], [0.299, 0.587, 0.144])
            image = image.astype(int)
    return(image)

# print(load_im("bird-1.bmp","images_Kimia216"))
# a = load_im("retine.png", grey=True)
# print(np.shape(a))
# load_im("cornee.png", grey=True)

def plot_im(images, legend = 0, binary = False):
    """plot the images in a window.
    The images are in a list"""
    i = len(images)
    fig, axs = plt.subplots(nrows = 1, ncols = i, figsize = (30,10))
    for j in range(i):
        ax = plt.subplot(1,i, j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if type(legend) == list:
            if len(legend) == i:
                plt.title(legend[j], fontsize=24)
            else:
                return("Error: Wrong number of elements in the legend")
        if binary:
            plt.imshow(images[j],  cmap = 'gray')
        else:
            plt.imshow(images[j])
    plt.show()
    return(fig)
    

# im1 = load_im("retine.png")
# im2 = load_im("muscle.jpg")
# im3 = load_im("cellules_cornee.jpg")
# plot_im([im1, im2, im3, im1],["legend", "legend", "legend", "legend"])

def plot_hist(images, legend = 0, y = 0):
    """plot the histograms of the grey images in a window
    The images are in a list"""
    i = len(images)
    if i<3:
        fig, axs = plt.subplots(nrows = 1, ncols = i, figsize = (15,10))
        for j in range(i):
            ax = plt.subplot(1, i, j+1)
            plt.xlim(0,256)
            if y != 0:
                plt.ylim(0, y)
            if type(legend) == list:
                if len(legend) == i:
                    plt.title(legend[j])
                else:
                    return("Error: Wrong number of elements in the legend")
            plt.hist(images[j].flatten(),256)
    else:
        fig, axs = plt.subplots(nrows = int((i+2)/3), ncols = 3, figsize = (15,10))
        for j in range(i):
            ax = plt.subplot(int((i+2)/3), 3, j+1)
            plt.xlim(0,256)
            if y != 0:
                plt.ylim(0, y)
            if type(legend) == list:
                if len(legend) == i:
                    plt.title(legend[j])
                else:
                    return("Error: Wrong number of elements in the legend")
            plt.hist(images[j].flatten(),256)
    plt.show()
    return(fig)


# =============================================================================
# TP2
# =============================================================================

def plot_im_hist(images, legend = 0, binary = False, y = 0):
    """Plot the images and their histogram under
    The images are in a list
    y : Sup boundary"""
    i = len(images)
    fig, axs = plt.subplots(nrows = 2, ncols = i, figsize = (30,9))
    for j in range(i):
        # Plot the image
        ax = plt.subplot(2,i, j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if type(legend) == list:
            if len(legend) == i:
                plt.title(legend[j], fontsize=24)
            else:
                return("Error: Wrong number of elements in the legend")
        if binary:
            plt.imshow(images[j],  cmap = 'gray')
        else:
            plt.imshow(images[j])
        # Plot the hist under
    for j in range(i):
        ax = plt.subplot(2, i,  i+j+1)
        plt.xlim(0,256)
        if y != 0:
            plt.ylim(0, y)
        if type(legend) == list:
            if len(legend) == i:
                plt.title(legend[j], fontsize=24)
            else:
                return("Error: Wrong number of elements in the legend")
        plt.hist(images[j].flatten(),256)
    plt.show()
    return(fig)

# =============================================================================
# TP 3
# =============================================================================

def bit_image(image):
    mmax = np.max(image)
    mmin = np.min(image)
    if (mmax == mmin):
        image=0
    else :
        image = 255*(image-mmin)/(mmax-mmin)
    return(image)

# Displays spectrum and phase in an image ( grayscale )
def plot_pectrumPhase(image, fourier_2D):
    """Return the image with the spectrum from a greyscale image :
         - log(1+ FFT)
         - phase 
         """
    
    phase = np.angle(fourier_2D)
    ampli = np.log(1 + abs(fourier_2D))
    phase = bit_image(phase)
    
    fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (15,10))
    # Plot the image
    ax = plt.subplot(1,3, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Vanilla image", fontsize=24)
    plt.imshow(image,  cmap = 'gray')

    # Plot the amplitude
    ax = plt.subplot(1,3,2)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Fourier amplitude", fontsize=24)
    plt.imshow( ampli, plt.cm.gray )
    
    # Plot the phase
    ax = plt.subplot(1,3,3)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Fourier phase", fontsize=24)
    plt.imshow(phase , plt.cm.gray )
    
    plt.plot()
    return(fig)







